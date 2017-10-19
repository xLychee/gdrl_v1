# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience, UpdatedExperience
from collections import deque


class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        #self.num_actions = self.env.get_num_actions()
        self.num_actions = Config.NUM_ENLARGED_ACTIONS
        #self.actions = Config.ENLARGED_ACTION_SET

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

        #### My Part
        self.epsilon = 0.0
        self.epsilon_decay = np.random.choice([0.99,0.995,0.95])
        self.epsilon_min = 0.1

        self.action_sequence = deque(maxlen=2)

    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, value, done):
        reward_sum = 0 if done else value
        return_list = []
        if done:
            for acs in Config.ENLARGED_ACTION_SET:
                if len(acs)>1 and acs[0] == experiences[-1].action:
                    action_index = Config.ACTION_INDEX_MAP[acs]
                    r = np.clip(experiences[-1].reward, Config.REWARD_MIN, Config.REWARD_MAX)
                    uexp = UpdatedExperience(experiences[-1].state, action_index, experiences[-1].prediction, r)
                    return_list.append(uexp)
                    #print("done:",acs, action_index, experiences[-1].prediction, r)
        action_sequence = ()
        for t in reversed(range(0, len(experiences))):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            action = experiences[t].action
            action_sequence = (action,) + action_sequence
            if action_sequence not in Config.ENLARGED_ACTION_SET:
                break
            action_index = Config.ACTION_INDEX_MAP[action_sequence]
            reward_sum = discount_factor * reward_sum + r
            uexp = UpdatedExperience(experiences[t].state, action_index, experiences[t].prediction, reward_sum)
            return_list.append(uexp)
            if  np.random.rand() < 0.0001:
                print(action_sequence, action_index, experiences[t].prediction, reward_sum)
        return return_list

    def convert_data(self, updated_experiences):
        x_ = np.array([exp.state for exp in updated_experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action_index for exp in updated_experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in updated_experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if self.action_sequence:
            return self.action_sequence.popleft()
        if Config.PLAY_MODE:
            action_index = np.argmax(prediction)
            action_set = Config.ACTION_INDEX_MAP[action_index]
            for a in action_set:
                self.action_sequence.append(a)
        else:
            if np.random.rand() <= self.epsilon:
                self.action_sequence.append(np.random.choice(Config.BASIC_ACTION_SET))
            else:
                action_index = np.random.choice(range(self.num_actions), p=prediction)
                action_set = Config.ACTION_INDEX_MAP[action_index]
                if not isinstance(action_set, tuple):
                    print(action_index, action_set)
                assert isinstance(action_set, tuple)
                for a in action_set:
                    self.action_sequence.append(a)
        if np.random.rand()<0.0001:
            print("epsilon: {}, action sequence: {}".format(self.epsilon,self.action_sequence))
        return self.action_sequence.popleft()

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        experience_queue = deque(maxlen=10)
        updated_exps = []

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step(0)  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            reward, done = self.env.step(action)
            reward_sum += reward
            exp = Experience(self.env.previous_state, action, prediction, reward, done)
            #experiences.append(exp)
            experience_queue.append(exp)
            updated_exps += ProcessAgent._accumulate_rewards(experience_queue, self.discount_factor, value, done)

            if (done or time_count == Config.TIME_MAX) and updated_exps:
                #terminal_reward = 0 if done else value
                #updated_exps = ProcessAgent._accumulate_rewards(experiences, self.discount_factor, terminal_reward)
                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                #experiences = [experiences[-1]]
                reward_sum = 0.0
                updated_exps = []

            time_count += 1
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
