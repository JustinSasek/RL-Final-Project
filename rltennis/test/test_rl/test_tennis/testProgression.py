from rl.tennis.discreteTennis import DiscreteTennis
from rl.tennis.behaviorLearnable import ShotSequenceFactory, LearnableTennisBehavior, Linear1xProgression, LinearRandom1xProgression, \
    LinearRandom1xEnd2xProgression
from rl.util.rtconfig import tc_runtime_configure
from copy import deepcopy

import unittest
import logging
import os
import sys

class ProgressionTest(unittest.TestCase):
    _logger = logging.getLogger("test.test_rl.tennis")
    _LOG_TAG = {"lgmod": "ProgCodec"}
    
    @classmethod
    def setUpClass(cls):
        # Configure logging:
        currDir=os.path.dirname(os.path.abspath(__file__))
        
        fac = ShotSequenceFactory()
        
        fac.register_seq(Linear1xProgression.NAME, 
            lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: \
            Linear1xProgression(receiver, start_at, nav_dir, step_delta, count_step, upto))
        
        fac.register_seq(LinearRandom1xEnd2xProgression.NAME, 
            lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: \
            LinearRandom1xEnd2xProgression(behavior, receiver, start_at, nav_dir, step_delta, count_step, upto))
        
        fac.register_seq(LinearRandom1xProgression.NAME, 
            lambda behavior, receiver, start_at, nav_dir, step_delta, count_step, upto: \
            LinearRandom1xProgression(behavior, receiver, start_at, nav_dir, step_delta, count_step, upto))
        
        cls.prog_factory = fac
        cls.behavior = LearnableTennisBehavior()
        tc_runtime_configure({
            "logConfig": currDir  + "/../../../logConfig.json"
        })
     
    def _assert_history(self, msg, start_at, history, step_delta, multiplier = 1, min_len = None):
        prev_shot_at = start_at
        delta_x = abs(step_delta[0])
        delta_y = abs(step_delta[1])
        count = 0
        
        if min_len is not None and len(history) < min_len:
            raise self.failureException("{}: Insufficient history items, expected={}, got={}".format(msg, min_len, len(history)))
        
        for shot_at in history:
            count = count + 1
            
            if (shot_at[0] < 0.0 or shot_at[0] > 1.0 or shot_at[1] < 0.0 or shot_at[1] > 1.0):
                raise self.failureException("{}: Invalid item [{}] = {}".format(msg, count, shot_at))
            
            step_x = abs(shot_at[0] - prev_shot_at[0])
            step_y = abs(shot_at[1] - prev_shot_at[1])
            
            diff_x = abs(step_x - delta_x)
            diff_y = abs(step_y - delta_y)
            
            multiplier_x = 1 if diff_x == 0 else (step_x/delta_x if delta_x > 0 else sys.maxsize)
            multiplier_y = 1 if diff_y == 0 else (step_y/delta_y if delta_y > 0 else sys.maxsize)
            
            if multiplier_x <= multiplier and multiplier_y <= multiplier:
                prev_shot_at = shot_at
                continue
            
            raise self.failureException("{}: item [{}] = {} is invalid w.r.t. {}".format(msg, count, shot_at, prev_shot_at))
        
    # @unittest.skip
    def testFront1x(self):
        self.run_test("front1x", (0.625, 0), (0.5, 0), Linear1xProgression.NAME, DiscreteTennis.DIR_FRONT, 1, 12)
    
    # @unittest.skip
    def testFrontRandom1x(self):
        self.run_test("frontRandom1x", (0.625, 0), (0.5, 0), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_FRONT, 1, None)
        
    # @unittest.skip
    def testFrontEnd2x(self):
        self.run_test("frontEnd2x", (0.625, 0), (0.5, 0), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_FRONT, 2, None)
        
    # @unittest.skip
    def testLeft1x(self):
        self.run_test("left1x", (0.625, 0), (0.5, 0), Linear1xProgression.NAME, DiscreteTennis.DIR_LEFT, 1, 8)
    
    # @unittest.skip
    def testLeftRandom1x(self):
        self.run_test("leftRandom1x", (0.625, 0), (0.5, 0), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_LEFT, 1, None)
        
    # @unittest.skip
    def testLeftEnd2x(self):
        self.run_test("leftEnd2x", (0.625, 0), (0.5, 0), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_LEFT, 2, None)
        
    # @unittest.skip
    def testBack1x(self):
        self.run_test("testBack1x", (0.625, 0), (0.5, 0.875), Linear1xProgression.NAME, DiscreteTennis.DIR_BACK, 1, 8)
    
    # @unittest.skip
    def testBackRandom1x(self):
        self.run_test("leftBack1x", (0.625, 0), (0.5, 0.875), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_BACK, 1, None)
        
    # @unittest.skip
    def testBackEnd2x(self):
        self.run_test("backEnd2x", (0.625, 0), (0.5, 0.875), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_BACK, 2, None)
        
    # @unittest.skip
    def testRight1x(self):
        self.run_test("right1x", (0.625, 0), (0.5, 0), Linear1xProgression.NAME, DiscreteTennis.DIR_RIGHT, 1, 8)
    
    # @unittest.skip
    def testRightRandom1x(self):
        self.run_test("rightRandom1x", (0.625, 0), (0.5, 0), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_RIGHT, 1, None)
        
    # @unittest.skip
    def testRightEnd2x(self):
        self.run_test("rightEnd2x", (0.625, 0), (0.5, 0), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_RIGHT, 2, None)
    
    # @unittest.skip
    def testFrontLeft1x(self):
        self.run_test("frontLeft1x", (0.625, 0), (0.5, 0), Linear1xProgression.NAME, DiscreteTennis.DIR_FRONTLEFT, 1, 8)
    
    # @unittest.skip
    def testFrontLeftRandom1x(self):
        self.run_test("frontLeftRandom1x", (0.625, 0), (0.5, 0), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_FRONTLEFT, 1, None)
        
    # @unittest.skip
    def testFrontLeftEnd2x(self):
        self.run_test("frontLeftEnd2x", (0.625, 0), (0.5, 0), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_FRONTLEFT, 2, None)
        
    # @unittest.skip
    def testBackLeft1x(self):
        self.run_test("backLeft1x", (0.625, 0), (0.5, 0.875), Linear1xProgression.NAME, DiscreteTennis.DIR_BACKLEFT, 1, 8)
    
    # @unittest.skip
    def testBackLeftRandom1x(self):
        self.run_test("backLeftRandom1x", (0.625, 0), (0.5, 0.875), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_BACKLEFT, 1, None)
        
    # @unittest.skip
    def testBackLeftEnd2x(self):
        self.run_test("backLeftEnd2x", (0.625, 0), (0.5, 0.875), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_BACKLEFT, 2, None)
        
    # @unittest.skip
    def testBackRight1x(self):
        self.run_test("backRight1x", (0.625, 0), (0.5, 0.875), Linear1xProgression.NAME, DiscreteTennis.DIR_BACKRIGHT, 1, 8)
    
    # @unittest.skip
    def testBackRightRandom1x(self):
        self.run_test("backRightRandom1x", (0.625, 0), (0.5, 0.875), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_BACKRIGHT, 1, None)
        
    # @unittest.skip
    def testBackRightEnd2x(self):
        self.run_test("backRightEnd2x", (0.625, 0), (0.5, 0.875), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_BACKRIGHT, 2, None)
    
    # @unittest.skip
    def testFrontRight1x(self):
        self.run_test("backRight1x", (0.625, 0), (0.5, 0), Linear1xProgression.NAME, DiscreteTennis.DIR_FRONTRIGHT, 1, 8)
    
    # @unittest.skip
    def testFrontRightRandom1x(self):
        self.run_test("backRightRandom1x", (0.625, 0), (0.5, 0), LinearRandom1xProgression.NAME, DiscreteTennis.DIR_FRONTRIGHT, 1, None)
        
    # @unittest.skip
    def testFrontRightEnd2x(self):
        self.run_test("backRightEnd2x", (0.625, 0), (0.5, 0), LinearRandom1xEnd2xProgression.NAME, DiscreteTennis.DIR_FRONTRIGHT, 2, None)
            
    def run_test(self, test_name, system_at, player_at, index, nav_dir, multipler, num_history = None):
        test_seq = self.prog_factory.build_seq(index, self.behavior, DiscreteTennis.PLAYER, 
            player_at, nav_dir)
        test_seq.set_log_history(True)
        req_step_delta = deepcopy(test_seq._step_delta)
        
        has_next = True
        while(has_next):
            next_target = test_seq.shot_target(system_at, player_at)
            if next_target is None:
                has_next = False
            
        if (self._logger.isEnabledFor(logging.INFO)):
            self._logger.info("{} Seq: {}: {}".format(test_name, test_seq, test_seq._history))
        self._assert_history(test_name, player_at, test_seq._history, req_step_delta, multipler, num_history)
        
if __name__ == '__main__':
    unittest.main()