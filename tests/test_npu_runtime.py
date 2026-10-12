# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 yolofast contributors
"""Verify conservative capability discovery and compiler evidence handling."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from inspect_npu_runtime import discover_bdfs, supported_reports, parse_partitions, summarize_allocation
from audit_npu_compile import audit
from npu_split import parser, profile_summary


class RuntimeEvidence(unittest.TestCase):
    def test_bdf_discovery_does_not_assume_old_device(self):
        self.assertEqual(discover_bdfs('NPU [0000:ab:12.3] NPU [0000:ab:12.3]'), ['0000:ab:12.3'])
        self.assertEqual(discover_bdfs('host-only report Topology 6x8'), [])
        self.assertEqual(discover_bdfs('0000:c6:00.1 0000:ab:12.3'), ['0000:ab:12.3','0000:c6:00.1'])

    def test_report_names_are_not_substring_matches(self):
        self.assertEqual(supported_reports('Valid reports: host'), [])
        self.assertEqual(supported_reports('Valid: aieshim, aiemem, aie-partitions'), ['aie-partitions','aiemem','aieshim'])

    def test_mismatched_saved_setting_is_not_called_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cache=root/'cache/key'; cache.mkdir(parents=True)
            context=cache/'context.json'
            context.write_text(json.dumps(dict(config=dict(sessionOptions={'opt_level':'3'},target='X2',passes=[dict(name='DPU',passDpuParam=dict(xcompilerAttrs={'opt_level':{'uintValue':0}}))]))))
            (cache/'compiled.xmodel').write_bytes(b'xmodel')
            (root/'run.log').write_text('PDI Swap times: 212\n')
            row=audit(context)
            self.assertEqual(row['requested']['opt_level'],'3')
            self.assertEqual(row['serialized_passes'][0]['opt_level'],{'uintValue':0})
            self.assertIsNone(row['effective_opt_level'])
            self.assertEqual(row['pdi'][0]['count'],212)
            self.assertEqual(row['artifacts'][0]['bytes'],6)

    def test_partition_schema_and_pid_scoped_counters(self):
        def payload(n):
            return json.dumps({'devices':[{'aie_partitions':{'partitions':[
                {'start_col':'0','num_cols':'8','hw_contexts':[
                    {'pid':'123','context_id':'4','status':'Active',
                     'command_submissions':str(n),'command_completions':str(n-1),'migrations':'0','errors':'0'},
                    {'pid':'999','context_id':'9','status':'Idle'}]}]}}]})
        observations=[dict(phase='steady',monotonic_ns=i,contexts=parse_partitions(payload(n))) for i,n in [(1,10),(2,20)]]
        row=summarize_allocation(observations,123,8)
        self.assertEqual(row['classification'],'all_8_columns_allocated')
        self.assertEqual(row['observed_max_columns'],8)
        self.assertEqual(row['submissions_completions_delta']['value'][0]['deltas']['command_submissions'],10)
        self.assertEqual(len(row['hw_contexts']['value']),4)
        self.assertEqual(summarize_allocation(observations,777,8)['classification'],'unknown')
        for observation in observations: observation['phase']='loaded-checkpoint-paused'
        self.assertEqual(summarize_allocation(observations,123,8)['classification'],'unknown')

    def test_text_table_and_changed_schema_fallback(self):
        text='Columns: [0, 1, 2, 3, 4, 5]\n|PID|Ctx ID|Status|Sub|Compl|Migr|Err|\n|123|4|Active|20|19|0|0|'
        contexts=parse_partitions(text)
        row=summarize_allocation([dict(phase='steady',monotonic_ns=1,contexts=contexts)],123,8)
        self.assertEqual(row['observed_max_columns'],6)
        self.assertEqual(row['classification'],'allocation_observed')
        self.assertEqual(parse_partitions('Topology 6x8'),[])
        self.assertEqual(parse_partitions('{"unrecognized": "schema"}'),[])

    def test_target_default_preserved_and_explicit_target_supported(self):
        self.assertIsNone(parser().parse_args(['--output','/tmp/test']).target)
        self.assertEqual(parser().parse_args(['--output','/tmp/test','--target','X1']).target,'X1')

    def test_profile_time_is_host_time_and_excludes_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'profile.json'
            path.write_text(json.dumps([dict(name='model_run',ph='X',ts=0,dur=99000),dict(name='model_run',ph='X',ts=100000,dur=12000)]))
            row=profile_summary(path,1,1)
            self.assertEqual(row['model_run_mean_ms'],12)
            self.assertIsNone(row['device_compute_ms'])
            self.assertIsNone(row['pdi_load_ms'])


if __name__=='__main__': unittest.main()
