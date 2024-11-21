import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
before_training = "before_training.mp4"
env = gym.make('Hopper-v2')
video = VideoRecorder(env, before_training)
env.reset()
for i in range(500):
    env.render()
    video.capture_frame()
    observation, reward, done, info = env.step(env.action_space.sample())
video.clse()
env.close()