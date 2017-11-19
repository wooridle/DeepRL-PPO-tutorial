import tensorflow as tf
import numpy as np
import gym
import roboschool


class Network(object):
    def __init__(self, env, scope, num_layers, num_units, obs_plc, act_plc, trainable=True):
        self.env = env
        self.observation_size = env.observation_space.shape[0]
        assert isinstance(env.action_space, gym.spaces.Box)
        self.action_size = env.action_space.shape[0]
        self.trainable = trainable

        self.scope = scope

        self.obs_place = obs_plc
        self.acts_place = act_plc

        self.p, self.v, self.logstd = self._build_network(num_layers=num_layers, num_units=num_units)
        self.act_op = self.action_sample()

    def _build_network(self, num_layers, num_units):
        with tf.variable_scope(self.scope):
            x = self.obs_place
            for i in range(num_layers):
                x = tf.layers.dense(x, units=num_units, activation=tf.nn.tanh, name="p_fc"+str(i),
                                    trainable=self.trainable)
            action = tf.layers.dense(x, units=self.action_size, activation=tf.tanh,
                                     name="p_fc"+str(num_layers), trainable=self.trainable)

            x = self.obs_place
            for i in range(num_layers):
                x = tf.layers.dense(x, units=num_units, activation=tf.nn.tanh, name="v_fc"+str(i),
                                    trainable=self.trainable)
            value = tf.layers.dense(x, units=1, activation=None, name="v_fc"+str(num_layers),
                                    trainable=self.trainable)

            logstd = tf.get_variable(name="logstd", shape=[self.action_size],
                                     initializer=tf.zeros_initializer)

        return action, value, logstd

    def action_sample(self):
        return self.p + tf.exp(self.logstd) * tf.random_normal(tf.shape(self.p))

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOAgent(object):
    def __init__(self, env):
        self.env = env

        ## hyperparameters
        self.learning_rate = 1e-4
        self.epochs = 10
        self.step_size = 3072
        self.gamma = 0.99
        self.lam = 0.95
        self.clip_param = 0.2
        self.batch_size = 64

        ## placeholders
        self.adv_place = tf.placeholder(shape=[None], dtype=tf.float32)
        self.return_place = tf.placeholder(shape=[None], dtype=tf.float32)

        self.obs_place = tf.placeholder(shape=[None, env.observation_space.shape[0]],
                                        name="ob", dtype=tf.float32)
        self.acts_place = tf.placeholder(shape=[None, env.action_space.shape[0]],
                                         name="ac", dtype=tf.float32)

        ## build network
        self.net = Network(env=self.env,
                           scope="pi",
                           num_layers=2,
                           num_units=128,
                           obs_plc=self.obs_place,
                           act_plc=self.acts_place)

        self.old_net = Network(env=self.env,
                               scope="old_pi",
                               num_layers=2,
                               num_units=128,
                               obs_plc=self.obs_place,
                               act_plc=self.acts_place,
                               trainable=False)

        # tensorflow operators
        self.assign_op = self.assign(self.net, self.old_net)
        self.ent, self.pol_loss, self.vf_loss, self.update_op = self.update()
        self.saver = tf.train.Saver()

    @staticmethod
    def logp(net):
        logp = -(0.5 * tf.reduce_sum(tf.square((net.acts_place - net.p) / tf.exp(net.logstd)), axis=-1) \
            + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(net.p)[-1]) \
            + tf.reduce_sum(net.logstd, axis=-1))

        return logp

    @staticmethod
    def entropy(net):
        ent = tf.reduce_sum(net.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return ent

    @staticmethod
    def assign(net, old_net):
        assign_op = []
        for (newv, oldv) in zip(net.get_variables(), old_net.get_variables()):
            assign_op.append(tf.assign(oldv, newv))

        return assign_op

    def traj_generator(self):
        t = 0
        action = env.action_space.sample()
        done = True
        ob = env.reset()

        cur_ep_return = 0
        cur_ep_length = 0
        ep_returns = []
        ep_lengths = []

        obs = np.array([ob for _ in range(self.step_size)])
        rewards = np.zeros(self.step_size, 'float32')
        values = np.zeros(self.step_size, 'float32')
        dones = np.zeros(self.step_size, 'int32')
        actions = np.array([action for _ in range(self.step_size)])
        prevactions = actions.copy()

        while True:
            prevaction = action
            action, value = self.act(ob)
            #print(value)
            if t > 0 and t % self.step_size == 0:
                yield {"ob": obs, "reward":rewards, "value": values,
                       "done": dones, "action": actions, "prevaction": prevactions,
                       "nextvalue": value*(1 - done), "ep_returns": ep_returns,
                       "ep_lengths": ep_lengths}

                ep_returns = []
                ep_lengths = []

            i = t % self.step_size
            obs[i] = ob
            values[i] = value
            dones[i] = done
            actions[i] = action[0]
            prevactions[i] = prevaction

            ob, reward, done, _ = env.step(action[0])
            rewards[i] = reward

            cur_ep_return += reward
            cur_ep_length += 1

            if done:
                print("Reward: {}".format(cur_ep_return))
                ep_returns.append(cur_ep_return)
                ep_lengths.append(cur_ep_length)
                cur_ep_return = 0
                cur_ep_length = 0
                ob = env.reset()
            t += 1

    def act(self, ob):
        action, value = tf.get_default_session().run([self.net.act_op, self.net.v], feed_dict={
            self.net.obs_place: ob[None]
        })
        return action, value

    def run(self):
        traj_gen = self.traj_generator()
        iteration = 0

        for _ in range(100000):
            iteration += 1
            print("\n================= iteration {} =================".format(iteration))
            traj = traj_gen.__next__()
            self.add_vtarg_and_adv(traj)

            tf.get_default_session().run(self.assign_op)

            traj["advantage"] = (traj["advantage"]-np.mean(traj["advantage"]))/np.std(traj["advantage"])

            len = int(self.step_size / self.batch_size)
            for _ in range(self.epochs):
                vf_loss = 0
                pol_loss = 0
                entropy = 0
                for i in range(len):
                    cur = i*self.batch_size
                    *step_losses, _ = tf.get_default_session().run([self.ent, self.vf_loss, self.pol_loss, self.update_op],feed_dict = {self.obs_place: traj["ob"][cur:cur+self.batch_size],
                                                       self.acts_place: traj["action"][cur:cur+self.batch_size],
                                                       self.adv_place: traj["advantage"][cur:cur+self.batch_size],
                                                       self.return_place: traj["return"][cur:cur+self.batch_size]})

                    entropy += step_losses[0]/len
                    vf_loss += step_losses[1]/len
                    pol_loss += step_losses[2]/len
                print("vf_loss: {:.5f}, pol_loss: {:.5f}, entorpy: {:.5f}".format(vf_loss, pol_loss, entropy))

            if iteration % 10 == 0:
                self.save_model('./model/Humanoid')

    def update(self):
        ent = self.entropy(self.net)
        ratio = tf.exp(self.logp(self.net) - tf.stop_gradient(self.logp(self.old_net)))
        surr1 = ratio * self.adv_place
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place

        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
        vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place))

        total_loss = pol_surr + 10*vf_loss

        update_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        return ent, pol_surr, vf_loss, update_op

    def add_vtarg_and_adv(self, traj):
        done = np.append(traj["done"], 0)
        value = np.append(traj["value"], traj["nextvalue"])
        T = len(traj["reward"])
        traj["advantage"] = gaelam = np.empty(T, 'float32')
        reward = traj["reward"]
        lastgaelam = 0

        for t in reversed(range(T)):
            nonterminal = 1 - done[t+1]
            delta = reward[t] + self.gamma * value[t+1] * nonterminal - value[t]
            gaelam[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        traj["return"] = traj["advantage"] + traj["value"]

    def save_model(self, model_path):
        self.saver.save(tf.get_default_session(), model_path)
        print("model saved")

    def restore_model(self, model_path):
        self.saver.restore(tf.get_default_session(), model_path)
        print("model restored")


if __name__ == "__main__":
    env = gym.make("RoboschoolHumanoid-v1")
    sess = tf.InteractiveSession()
    ppo = PPOAgent(env)
    tf.get_default_session().run(tf.global_variables_initializer())
    ppo.restore_model("./model/Humanoid")
    ppo.run()

    env.close()
