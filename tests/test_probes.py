"""Check probe geometry, temporal inputs and split-aware metrics."""
import unittest

import numpy as np
import torch

from ssl_levjepa import probes


class ProbeTests(unittest.TestCase):
    def test_motion_regimes_have_disjoint_boundaries(self):
        target = np.array([[0,0,.5,.5],[.05,0,.5,.5],[.5,0,.5,.5],[2,0,.5,.5]],dtype=np.float32)
        clips = [{"split":"test","match_id":"heldout","scenario_family":"unit"} for _ in target]
        result = probes.grouped_metrics(target,target.copy(),clips)
        self.assertEqual([v["n"] for v in result["test"]["motion"].values()],[1,1,1,1])
        self.assertEqual(result["test"]["aggregate"]["velocity_rmse_mps"],0)
        self.assertEqual(result["test"]["aggregate"]["within_8px"],1)
        self.assertEqual(result["train"]["aggregate"]["n"],0)

    def test_head_preserves_supervised_encoder_gradients(self):
        feature = torch.zeros(2,2,192,14,14,requires_grad=True)
        clips = [{"dt_last_sec":.03},{"dt_last_sec":.04}]
        value = probes._head_input(feature,np.arange(2),clips,device="cpu")
        self.assertEqual(tuple(value.shape),(2,384,14,14))
        value[:,192:].sum().backward()
        self.assertGreater(float(feature.grad[:,1].sum()),0)
        self.assertLess(float(feature.grad[:,0].sum()),0)

    def test_temporal_difference_uses_actual_frame_interval(self):
        feature = np.zeros((2,2,196,192),dtype=np.float16)
        feature[:,1] = 2
        clips = [{"dt_last_sec":.03},{"dt_last_sec":.04}]
        center,scale = torch.ones(192),torch.ones(192)*2
        value = probes._head_input(feature,np.arange(2),clips,device="cpu",normalization=(center,scale))
        self.assertAlmostEqual(float(value[0,0,0,0]),.5)
        self.assertAlmostEqual(float(value[0,192,0,0]),1/.9,places=5)
        self.assertAlmostEqual(float(value[1,192,0,0]),1/1.2,places=5)

    def test_decoding_uses_predicted_location(self):
        maps = torch.full((1,5,14,14),0.0)
        maps[:,0] = -100
        maps[0,0,4,3] = 100
        maps[0,3,4,3] = 2
        maps[0,4,4,3] = -1
        mean = torch.tensor([.5,.25,0,0])
        scale = torch.tensor([2.,3.,1.,1.])
        prediction = probes._head_decode(maps,mean,scale)
        np.testing.assert_allclose(prediction.numpy(),[[4.5,-2.75,3.5/14,4.5/14]],rtol=1e-6)
        target = prediction.detach()
        loss = probes._head_loss(maps,target,mean,scale)
        self.assertAlmostEqual(float(loss),0,places=6)

    def test_reported_rmse_is_component_wise(self):
        target = np.zeros((1,4),dtype=np.float32)
        prediction = np.array([[3,4,1/256,2/256]],dtype=np.float32)
        result = probes.metrics(target,prediction)
        self.assertAlmostEqual(result["velocity_rmse_mps"],np.sqrt(12.5),places=6)
        self.assertAlmostEqual(result["velocity_epe_mps"],5,places=6)
        self.assertAlmostEqual(result["localization_rmse_px"],np.sqrt(2.5),places=6)


if __name__=="__main__":
    unittest.main()
