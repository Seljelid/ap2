
class ActorNetwork(object):
    """
    Description of class.
    """
    
    def actor_network(self):
        
        
    def train():
        

    def predict():
        
    
    def predict_target():
        
    
    def update_target_network():
        self.sess.run(self.update_target_network_params)
    
    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    
class CriticNetwork(object):
    """
    Description of class
    """
    
    def critic_network(self):
        
        
    def train():
        
        
    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
        
    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })
        
    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })
    
    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
        
def build_summaries():
episode_reward = tf.Variable(0.)
tf.summary.scalar("Reward", episode_reward)
episode_ave_max_q = tf.Variable(0.)
tf.summary.scalar("Qmax Value", episode_ave_max_q)

summary_vars = [episode_reward, episode_ave_max_q]
summary_ops = tf.summary.merge_all()

return summary_ops, summary_vars

def train(sess, env, actor, critic):
    
    summary_ops, summary_vars = build_summaries()
    
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    
    actor.update_target_network()
    critic.update_target_network()
    
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)
    
    for i in xrange(MAX_EPISODES):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in xrange(MAX_EP_STEPS):

            if RENDER_ENV:
                env.render()

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in xrange(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print '| Reward: %.2i' % int(ep_reward), " | Episode", i, \
                    '| Qmax: %.4f' % (ep_ave_max_q / float(j))

                break

def main():
    with tf.Session() as sess:
    
        actor = ActorNetwork()
        
        critic = CriticNetwork()
        
        train(sess, actor, critic)
    

if __name__ == '__main__':
    tf.app.run()
    
    
    
    
    
    
    
    
    
    

