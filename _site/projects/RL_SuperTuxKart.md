#**EC418 Fall 2022 Final Project: PPO SuperTuxKart**

**Goal:** Get an RL agent to learn how to play SuperTuxKart by getting as far as it can on a given track.

**Approach:** Proximal Policy Optimization (PPO)

#####**Installing Dependencies**


```python
%pip install -U PySuperTuxKart

# Hide Output with '> /dev/null 2>&1'
!apt-get update > /dev/null 2>&1
!apt-get install -y xvfb python-opengl ffmpeg xvfbwrapper cmake > /dev/null 2>&1
!pip install pyvirtualdisplay > /dev/null 2>&1
!pip install pyglet==1.5.27 > /dev/null 2>&1 # newer versions deprecate functions we need
!pip install ez_setup > /dev/null 2>&1
!pip install gym==0.21.0 > /dev/null 2>&1
!pip install -U ray > /dev/null 2>&1
!pip install imageio==2.4.1 > /dev/null 2>&1
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting PySuperTuxKart
      Downloading PySuperTuxKart-1.1.2-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (4.4 MB)
    [K     |████████████████████████████████| 4.4 MB 4.2 MB/s 
    [?25hCollecting PySuperTuxKartData
      Downloading PySuperTuxKartData-1.0.0.tar.gz (2.6 kB)
      Installing build dependencies ... [?25l[?25hdone
      Getting requirements to build wheel ... [?25l[?25hdone
        Preparing wheel metadata ... [?25l[?25hdone
    Requirement already satisfied: requests in /usr/local/lib/python3.8/dist-packages (from PySuperTuxKartData->PySuperTuxKart) (2.23.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests->PySuperTuxKartData->PySuperTuxKart) (2022.9.24)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests->PySuperTuxKartData->PySuperTuxKart) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.8/dist-packages (from requests->PySuperTuxKartData->PySuperTuxKart) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests->PySuperTuxKartData->PySuperTuxKart) (2.10)
    Building wheels for collected packages: PySuperTuxKartData
      Building wheel for PySuperTuxKartData (PEP 517) ... [?25l[?25hdone
      Created wheel for PySuperTuxKartData: filename=PySuperTuxKartData-1.0.0-py3-none-any.whl size=620700391 sha256=ccba3ca39b772aef18a40074f405d853bacbb8c7d5f4b80350740fcf6cd33357
      Stored in directory: /root/.cache/pip/wheels/58/6b/b7/6b714bf378c4149d2ec72f61ef186e92902085c3cf43ac531e
    Successfully built PySuperTuxKartData
    Installing collected packages: PySuperTuxKartData, PySuperTuxKart
    Successfully installed PySuperTuxKart-1.1.2 PySuperTuxKartData-1.0.0


#####**Mounting to Drive for Saving Results**


```python
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
%cd /content/drive/MyDrive/
!ls
```

    Mounted at /content/drive/
    /content/drive/MyDrive
     500_epoch.mp4	    controller.th	     Pictures	   result3.mp4
     Archive	    hacienda_controller.th   result1.mp4   snowtuxpeak.mp4
    'Colab Notebooks'   hacienda.mp4	     result2.mp4   __temp__.mp4


####**Utils**

#####prepare_video function based on WandB's video data source:


```python
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import display

def prepare_video(video: "np.ndarray") -> "np.ndarray":
        """This logic was mostly taken from tensorboardX"""
        if video.ndim < 4:
            raise ValueError(
                "Video must be atleast 4 dimensions: time, channels, height, width"
            )
        if video.ndim == 4:
            video = video.reshape(1, *video.shape)
        b, t, c, h, w = video.shape

        if video.dtype != np.uint8:
            #logging.warning("Converting video data to uint8")
            video = video.astype(np.uint8)

        def is_power2(num: int) -> bool:
            return num != 0 and ((num & (num - 1)) == 0)

        # pad to nearest power of 2, all at once
        if not is_power2(video.shape[0]):
            len_addition = int(2 ** video.shape[0].bit_length() - video.shape[0])
            video = np.concatenate(
                (video, np.zeros(shape=(len_addition, t, c, h, w))), axis=0
            )

        n_rows = 2 ** ((b.bit_length() - 1) // 2)
        n_cols = video.shape[0] // n_rows

        video = np.reshape(video, newshape=(n_rows, n_cols, t, c, h, w))
        video = np.transpose(video, axes=(2, 0, 4, 1, 5, 3))
        video = np.reshape(video, newshape=(t, n_rows * h, n_cols * w, c))
        return video
```

    Imageio: 'ffmpeg-linux64-v3.3.1' was not found on your computer; downloading it now.
    Try 1. Download from https://github.com/imageio/imageio-binaries/raw/master/ffmpeg/ffmpeg-linux64-v3.3.1 (43.8 MB)
    Downloading: 8192/45929032 bytes (0.0%)2678784/45929032 bytes (5.8%)6946816/45929032 bytes (15.1%)11132928/45929032 bytes (24.2%)15499264/45929032 bytes (33.7%)19734528/45929032 bytes (43.0%)23945216/45929032 bytes (52.1%)28139520/45929032 bytes (61.3%)32243712/45929032 bytes (70.2%)36626432/45929032 bytes (79.7%)41000960/45929032 bytes (89.3%)45301760/45929032 bytes (98.6%)45929032/45929032 bytes (100.0%)
      Done
    File saved as /root/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1.


#####make_video function based on the rollout data


```python
import ray
from collections import deque
from PIL import Image, ImageDraw
import imageio
from IPython.display import Video
from IPython.display import display

def make_video(rollouts, m=4):
    videos = list()
    max_t = 0

    for rollout in rollouts:
        video = list()

        for data in rollout:
            video.append(data.s.transpose(2, 0, 1))

        videos.append(np.uint8(video))
        max_t = max(max_t, len(rollout))

    videos.sort(key=lambda x: x.shape[0], reverse=True)

    _, c, h, w = videos[0].shape
    full = np.zeros((max_t, c, h * m, w * m), dtype=np.uint8)

    # blocking up the videos to view multiple at the same time
    for i in range(m):
        for j in range(m):
            if i * m + j >= len(videos):
                continue
            n = videos[i * m + j].shape[0]

            full[:n, :, i * h: i * h + h, j * w: j * w + w] = videos[i * m + j]
            full[n:, :, i * h: i * h + h, j * w: j * w + w] = videos[i * m + j][-1][None]

    return full
```

#####simple geometry functions for getting distance for calc supertuxkart data in rollout


```python
def point_from_line(p, a, b):
    u = p - a
    u = np.float32([u[0], u[2]])
    v = b - a
    v = np.float32([v[0], v[2]])
    v_norm = v / np.linalg.norm(v)
    close = u.dot(v_norm) * v_norm
    return np.linalg.norm(u - close)

def get_distance(d_new, d, track_length):
    if abs(d_new - d) > 100:
        sign = float(d_new < d) * 2 - 1
        d_new, d = min(d_new, d), max(d_new, d)
        return sign * ((d_new - d) % track_length)
    return d_new - d
```

#**Reinforcement Learning for SuperTuxKart**

#####**Neural Network Declaration for actor & critic network**
**actor** - model performs the task of learning what action to take under state observation of the environment.

**critic** - evaluate if the action taken leads to a better state or not. Outputs a rating (Q-value) of action taken in the previous state.


```python
import numpy as np
import torch
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self, n_outputs):
        super().__init__()
        self.n_outputs = n_outputs

        # BatchNorm to help stabilize network during training [ref: https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739]
        self.norm = torch.nn.BatchNorm2d(1) # 2D since we want to look at the screenshots

        self.conv1 = torch.nn.Conv2d(1, 32,
                                     kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)

        self.fc4 = torch.nn.Linear(4*4*64, 512)
        self.fc5 = torch.nn.Linear(512, n_outputs)


    def forward(self, x):
        x = self.norm(x.mean(1, True))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1))) # reshaping tensor w/o copy
        x = self.fc5(x)
        return x

    def get_output_size(self):
        return self.n_outputs

```

#####**Discretized Policy**
Includes possible actions that the agent (actor) may take to drive the kart

**NOTE**: might need to change max acceleration

\[ ref: https://arxiv.org/pdf/1901.10500.pdf ]


```python
from typing import Dict
import pystk
import torch
import numpy as np


class BasePolicy:
    def __call__(self, m: Dict):
        raise NotImplementedError("__call__ in class BasePolicy")

class DiscretizedPolicy():
    def __init__(self, neural_net, n_actions, eps):
        self.n_actions = n_actions
        self.neural_net = neural_net
        self.neural_net.eval().cpu() # turn off BatchNorm layer for model eval && move tensor to CPU mem
        self.eps = eps # based on current epoch being iterated on eps = np.clip((1 - epoch / 100) * self.eps, 0.0, 1.0) on a higher level

    # function that enables the class to behave like a function when called
    def __call__(self, state, Q):
        # detach gradients attached to tensors
        with torch.no_grad():
            state = state.transpose(2, 0, 1)
            # flatten it out
            state = torch.FloatTensor(state).unsqueeze(0).cpu() # move to CPU mem
            # create a categorical distribution with event log probabilities (logits)
            m = torch.distributions.Categorical(logits=self.neural_net(state))

        if np.random.rand() < self.eps:
            action_indx = np.random.choice(list(range(self.n_actions))) # random exploration
        else:
            action_indx = m.sample().item() # sample a event distribution for action index

        # probability of action for a event state
        p = m.probs.squeeze()[action_indx]
        p_action = (1 - self.eps) * p + self.eps / (2**4)
        # 4 actions include: steer left Y|N, steer right Y|N, accelerate Y|N, drifting Y|N
        action = pystk.Action()

        # turn action_indx into a 4 digit binary that corresponds to which of the 4 main actions
        # and if we should do said action or not
        # 16 possible actions -> can be rep by 1111b
        binary = bin(action_indx).lstrip('0b').rjust(4, '0')

        # digit x___ determines if steer left (-1.0) -> NOTE: a little less of a sharp turn now
        # digit _x__ determines if steer right (1.0)
        # TODO: base this on Q like how acceleration is for more fidelity! -> maybe can handle higher accels?
        if (binary[0] == '1'):
            # steer to the left
            action.steer = -1.0 * np.clip(1 + int(binary[0] == '1') * 25.0 - Q, 0.0, 0.9)
        elif (binary[1] == '1'):
            # steer to the right
            action.steer = 1.0 * np.clip(1 + int(binary[1] == '1') * 25.0 - Q, 0.0, 0.9)
        #action.steer = int(binary[0] == '1') * -1.0 + int(binary[1] == '1') * 1.0 # OG

        # NOTE: the system usually will go with the max value though
        # digit __x_ determines acceleration value based on Q clipped to range [0.0, 0.5]
        # kind of arbitrarily calculated but varied by Q (high reward -> accelerate more)
        action.acceleration = np.clip(5 + int(binary[2] == '1') * 20.0 - Q, 0.0, 0.7)
        #action.acceleration = np.clip(5 + int(binary[2] == '1') * 20.0 - Q, 0, 0.5) # OG

        # digit ___x determines drift boolean value
        action.drift = bool(binary[3] == '1')
        # NOTE: for increasing performance, might want to action.brake = True before drifting

        return action, action_indx, p_action
```

####**PPO Implementation**
**PPO-clip** version is used here as described in the following references:

\[https://spinningup.openai.com/en/latest/algorithms/ppo.html ]

\[https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8 ]

\[https://arxiv.org/pdf/1707.06347.pdf ]

\[https://stats.stackexchange.com/questions/326608/why-is-ppo-able-to-train-for-multiple-epochs-on-the-same-minibatch ]

**Actor-Critic Framework**
\[https://tinyurl.com/435cyyn8 ]


```python
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class PPO(object):
  def __init__(self, batch_size, learn_r, iters, eps, gamma, clip, device, **kwargs): # needed for Ray trainer
    self.batch_size = batch_size
    self.learn_r = learn_r
    self.iters = iters
    self.eps = eps
    self.gamma = gamma
    self.clip = clip # the epsilon from algo step 6
    self.device = device
    self.n_actions = 2**4 # hardcoded from the DiscretizedPolicy, probably not best practices

    # actor model
    self.actor = Network(self.n_actions)
    self.actor.to(self.device) # mem format for desired device
    # optimizer for actor
    self.optimr_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learn_r)

    # critic model
    self.critic = Network(1) # only one output for criticizing actor i.e. TD error
    self.critic.to(self.device)
    # optimizer for critic
    self.optimr_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learn_r)

  # return the action, action_indx, p_action from the discretized policy based on actor model
  def get_policy(self, epoch):
    return DiscretizedPolicy(self.actor,
                             n_actions=self.n_actions,
                             eps=(np.clip((1 - epoch / 100) * self.eps, 0.0, 1.0)))


  # where the PPO learning magic is!
  def train(self, replay):
    # prepare the nns' losses
    losses_actor = list()
    losses_critic = list()

    # format/send nns to specified device mem
    self.actor.to(self.device)
    self.critic.to(self.device)

    # training the actor and critic models
    self.actor.train()
    self.critic.train()

    # main PPO-clip algorithm
    for iter in range(self.iters):
      # choose a random point in the replay buffer to analyze
      indices = np.random.choice(len(replay), self.batch_size)
      s, _, p_i, p_k, r, _, Q, done = replay[indices]

      # format/send tensors to specified device mem
      s   = torch.FloatTensor(s.transpose(0, 3, 1, 2)).to(self.device) # state (i.e. observed batch)
      p_i = torch.LongTensor(p_i).squeeze().to(self.device)  # current policy params
      p_k = torch.FloatTensor(p_k).squeeze().to(self.device) # prev_policy params
      Q   = torch.FloatTensor(Q).squeeze().to(self.device)   # qval

      # get current policy
      m = torch.distributions.Categorical(logits=self.actor(s))
      # get the policy ratio
      ratio_p = torch.exp(m.log_prob(p_i)) / p_k

      # get critic value
      V = self.critic(s).squeeze()

      # calculating advantage function A = Q - V
      A = Q - self.critic(s).squeeze()
      # normalize the advantages
      A = (A - A.mean()) / (A.std() + 1e-7) # the 1e-7 is there to avoid dividing by 0
      # lose the gradient portion
      A = A.detach()


      # the PPO-clip objective function i.e. step 6 of the algo
      objective = torch.min(ratio_p * A,
                            torch.clamp(ratio_p, 1 - self.clip, 1 + self.clip) * A)

      # param update losses
      # actor param loss
      loss_actor = -(objective + (1e-2 * m.entropy())).mean()

      # critic param loss i.e. step 7 of the algo (MSE suffices here)
      loss_critic = ((V - Q)**2).mean()

      # calculate gradients and perform backward propagation for networks
      loss_actor.backward()
      self.optimr_actor.step()
      self.optimr_actor.zero_grad()
      loss_critic.backward()
      self.optimr_critic.step()
      self.optimr_critic.zero_grad()

      # append params
      losses_actor.append(loss_actor.item())
      losses_critic.append(loss_critic.item())

    # return mean with the new update
    n_losses_actor = np.mean(losses_actor)
    n_losses_critic = np.mean(losses_critic)
    return np.mean(losses_actor), np.mean(losses_critic)
```

###**Replay Buffer**

adapted from PPO from scratch tutorial


```python
import collections

#       s, _, p_i, p_k, r, _, Q, done = replay[indices]
Data = collections.namedtuple('Data', 's a p_i p_k r s_p Q done')

# simple buffer type list
class Buffer(object):
    def __init__(self, val, max_size):
        self.val = val
        self.max_size = max_size
        self.len = 0
        self.position = 0
        self.buffer = np.zeros((self.max_size,) + self.val.shape, dtype=self.val.dtype)

    def add(self, val):
        self.buffer[self.position] = val.copy()
        self.position = (self.position + 1) % self.max_size
        self.len += 1

    def __getitem__(self, key):
        return self.buffer[key]

    def __len__(self):
        return min(self.max_size, self.len)


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffers = dict()
        self.max_size = max_size

    def get_info(self):
        return self.buffers, self.max_size

    def add(self, data):
        for key in data._fields:
            val = getattr(data, key)

            if key not in self.buffers:
                self.buffers[key] = Buffer(val, self.max_size)

            self.buffers[key].add(val)

    def __getitem__(self, idx):
        result = list()
        for key in Data._fields:
            result.append(self.buffers[key][idx])
        return result

    def __len__(self):
        lens = list()
        for _, val in self.buffers.items():
            lens.append(len(val))
        assert min(lens) == max(lens)
        return lens[0]
```

###**Rollout Data**

Rollout logic taken from the gym env reference and the original rollout code.

**NOTE:** This is where the video gets saved, change here to change video file name


```python
import ray
from collections import deque
from PIL import Image, ImageDraw
import imageio
from IPython.display import Video
from IPython.display import display
from typing import Dict


class Rollout(object):
    def __init__(self, track):
        config = pystk.GraphicsConfig.ld()
        config.screen_width = 64
        config.screen_height = 64
        #pystk.clean() # just in case you crash and couldn't terminate the program
        pystk.init(config)
        race_config = pystk.RaceConfig()
        race_config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
        race_config.track = track
        race_config.step_size = 0.1

        # pytux race setup
        self.race = pystk.Race(race_config)
        self.race.start()
        self.race.step()

        # pytux track start
        self.track = pystk.Track()
        self.track.update()

    def rollout(self,
                policy: BasePolicy,
                max_step: float=100,
                frame_skip: int=0,
                gamma: float=1.0):

        # always make sure to restart the track environment!
        self.race.restart()
        self.race.step(pystk.Action())
        self.track.update()

        state = pystk.WorldState()
        state.update()
        result = list()
        r_total = 0
        d = state.karts[0].distance_down_track
        s = np.array(self.race.render_data[0].image)
        G = 0
        off_track = deque(maxlen=20)
        traveled = deque(maxlen=50)

        for i in range(max_step):
            # Early termination
            # depends on being off the track and not moving
            if i > 20 and (np.median(traveled) < 0.05 or all(off_track)):
                break
            v = np.linalg.norm(state.karts[0].velocity)
            action, action_i, p_action = policy(s, v)
            if isinstance(action, pystk.Action):
                action_op = [action.steer, action.acceleration, action.drift]
            else:
                action_op = action
                action = pystk.Action()
                action.steer = action_op[0]
                action.acceleration = np.clip(action_op[1] - v, 0, np.inf)
                action.drift = action_op[2] > 0.5

            for j in range(1 + frame_skip):
                self.race.step(action)
                self.track.update()
                state = pystk.WorldState()
                state.update()

            s_p = np.array(self.race.render_data[0].image)
            d_new = min(state.karts[0].distance_down_track, d + 5.0)
            node_idx = np.searchsorted(
                    self.track.path_distance[:, 1],
                    d_new % self.track.path_distance[-1, 1]) % len(self.track.path_nodes)
            a_b = self.track.path_nodes[node_idx]
            distance = point_from_line(state.karts[0].location, a_b[0], a_b[1])
            distance_traveled = get_distance(d_new, d, self.track.path_distance[-1, 1])
            gain = distance_traveled if distance_traveled > 0 else 0
            mult = int(distance < 6.0)
            traveled.append(gain)
            off_track.append(distance > 6.0)

            r_total = max(r_total, d_new * mult)
            r = np.clip(0.5 * max(mult * gain, 0) + 0.5 * mult, -1.0, 1.0)

            # print when the kart finished the track (if it was able to)
            if np.isclose(state.karts[0].overall_distance / self.track.length, 1.0, atol=2e-3):
                print('Finished at: t=%d' % i)

            result.append(
                    Data(s.copy(),
                         np.float32(action_op),
                         np.uint8([action_i]),
                         np.float32([p_action]),
                         np.float32([r]),
                         s_p.copy(),
                         np.float32([np.nan]),
                         np.float32([0])))
            d = d_new
            s = s_p

        for i, data in enumerate(reversed(result)):
            G = data.r + gamma * G
            # collections.namedtuple('Data', 's a p_i p_k r s_p Q done')
            result[-(i + 1)] = Data(data.s,
                                    data.a,
                                    data.p_i,
                                    data.p_k,
                                    data.r,
                                    data.s_p,
                                    np.float32([G]),
                                    np.float32([i == 0]))

        return result[4:], (r_total / self.track.path_distance[-1, 1])


    def rollout_eval(self, policy: BasePolicy, max_step, frame_skip, gamma):
        COLAB_IMAGES = list()
        # always make sure to restart the track environment!
        self.race.restart()
        self.race.step(pystk.Action())
        self.track.update()

        state = pystk.WorldState()
        state.update()
        result = list()
        r_total = 0
        d = state.karts[0].distance_down_track
        s = np.array(self.race.render_data[0].image)
        G = 0
        off_track = deque(maxlen=20)
        traveled = deque(maxlen=50)

        for i in range(max_step):
            # Early termination
            # depends on being off the track and not moving
            if i > 20 and (np.median(traveled) < 0.05 or all(off_track)):
                break
            v = np.linalg.norm(state.karts[0].velocity)
            action, action_i, p_action = policy(s, v)
            if isinstance(action, pystk.Action):
                action_op = [action.steer, action.acceleration, action.drift]
            else:
                action_op = action
                action = pystk.Action()
                action.steer = action_op[0]
                action.acceleration = np.clip(action_op[1] - v, 0, np.inf)
                action.drift = action_op[2] > 0.5

            for j in range(1 + frame_skip):
                self.race.step(action)
                self.track.update()
                state = pystk.WorldState()
                state.update()

            image = Image.fromarray(self.race.render_data[0].image)
            COLAB_IMAGES.append(np.array(image))
            clip = ImageSequenceClip(COLAB_IMAGES, fps=15)
            filename = 'eval.mp4'
            clip.write_videofile(filename)
            clip.ipython_display(width = 256)


    def __del__(self):
        #self.race.stop()
        self.race = None
        self.track = None
        pystk.clean()


# makes Rollout compatible with Ray
@ray.remote(num_cpus=1, num_gpus=0.1)
class RayRollout(Rollout):
    pass

N_WORKERS_CPU = 8 # how many cpu for ray you specified in main
class RaySampler(object):
    def __init__(self, track='hacienda'):
        self.rollouts = [RayRollout.remote(track) for _ in range(N_WORKERS_CPU)]
        #self.rollouts_eval = RayRollout.remote(track) # Ray doesn't like having multiple ray.remote objects at once

    def get_samples(self, agent, max_frames=10000, max_step=500, gamma=1.0, frame_skip=0, **kwargs):
        total_frames = 0
        returns = list()
        video_rollouts = list()

        while total_frames <= max_frames:
            batch_ros = list()
            for rollout in self.rollouts:
                batch_ros.append(rollout.rollout.remote(agent,
                                                        max_step=max_step,
                                                        gamma=gamma,
                                                        frame_skip=frame_skip))
            batch_ros = ray.get(batch_ros)
            if len(video_rollouts) < 64:
                video_rollouts.extend([ro for ro, ret in batch_ros if len(ro) > 0])

            total_frames += sum(len(ro) * (frame_skip + 1) for ro, ret in batch_ros)
            returns.extend([ret for ro, ret in batch_ros])

            yield batch_ros

        print('Total Frames: %d' % (total_frames))
        print('Episodes: %d' % (len(returns)))
        print('Return: %.3f' % np.mean(returns))

        # try to use video_rollouts to show stuff
        print("videos:", make_video(video_rollouts).shape) # (time, channel, height, width)

        tensor = prepare_video(make_video(video_rollouts))
        clip = ImageSequenceClip(list(tensor), fps=15)
        filename = 'scotland.mp4'
        clip.write_videofile(filename)
        clip.ipython_display(width = 256) # TODO: figure out why/how to make this play in output whilst training

    # BROKEN
    def get_samples_eval(self, agent, max_frames=10000, max_step=500, gamma=1.0, frame_skip=0, **kwargs):
        total_frames = 0
        returns = list()
        video_rollouts = list()

        while total_frames <= max_frames:
            batch_ros = list()
            batch_ros.append(self.rollouts_eval.rollout.remote(agent,
                                                        max_step=max_step,
                                                        gamma=gamma,
                                                        frame_skip=frame_skip))
            #print('batch_rollouts: ', batch_ros)
            batch_ros = ray.get(batch_ros)
            if len(video_rollouts) < 64:
                video_rollouts.extend([ro for ro, ret in batch_ros if len(ro) > 0])

            total_frames += sum(len(ro) * (frame_skip + 1) for ro, ret in batch_ros)
            returns.extend([ret for ro, ret in batch_ros])

            yield batch_ros

        # try to use video_rollouts to show stuff
        print("videos:", make_video(video_rollouts).shape) # (time, channel, height, width)

        tensor = prepare_video(make_video(video_rollouts))
        clip = ImageSequenceClip(list(tensor), fps=15)
        filename = 'eval.mp4'
        clip.write_videofile(filename)
        clip.ipython_display(width = 256) # TODO: figure out why/how to make this play in output whilst training
```

#**MAIN**
Configs and actually learning!

Ray Ref:
\[https://docs.ray.io/en/latest/ray-observability/ray-metrics.html#ray-metrics ]


```python
import argparse
import pathlib
from os import path

def main(config):
    # Ray trainer
    trainer = {'ppo': PPO}[config['algorithm']](**config)

    sampler = RaySampler(config['track'])
    replay = ReplayBuffer(config['max_frames'])

    for epoch in range(config['max_epoch']+1):
        print("\033[1m" + 'EPOCH: ' + "\033[0m", epoch)
        for rollout_batch in sampler.get_samples(trainer.get_policy(epoch), **config):
            for rollout, _ in rollout_batch:
                for data in rollout:
                    replay.add(data)

        # metrics from ray trainer based on the current replay from the buffer
        # of which the replay buffer will be used by the PPO to train
        trainer.train(replay)

        # save final few epoch trained model (don't want to save all the premature ones)
        if epoch % 50 != 0:
          model_save_name = 'controller.th'
          path = F"/content/drive/MyDrive/{model_save_name}"
          torch.save(trainer.actor.state_dict(), path)

```

**NOTE:**
Besides the parameters listed here, the performance is also affected by the range of values for the actions in the discretized policy class. This is due to the discretization being rather gross. -> limiting max acceleration seems to help for some reason

**NOTE:**
Be sure to change the saved .mp4 file's name in the RaySampler class to whatever the track name is

**PPO Best Practices:**
\[https://github.com/llSourcell/Unity_ML_Agents/blob/master/docs/best-practices-ppo.md ]

Return should continue to increase per epoch if training is going well!


```python
config = {'algorithm': 'ppo', # letting ray trainer know which algo we wanna go with
          'track': 'scotland', # pytuxkart track name
          'frame_skip' : 1,
          'max_frames': 10000, # Essentially works as max buffer in our case, should be a multiple of batch_size (larger means more stable training apparently)
          'max_step' : 500, # how many steps of the simulation are run through training (more steps for more complex problems)
          'gamma': 0.9, # discount factor
          'max_epoch': 500, # number of passes through the whole dataset
          'device': torch.device('cuda'), # if no cuda, change to 'cpu'
          'batch_size': 512, # number of samples processed before the model is updated (power of 2)
          'iters': 100, # number of times a batch is passed through [10-100]
          'learn_r': 5e-4, # correspond to strength of each gradient descent update step
          'clip': 0.10, # acceptable threshold of divergence b/w old & new policies
          'eps': 0.10, # helps with exploration (higher #) vs exploitation (lower #)
          }

# checking prev. ray node status
# NOTE: if running for the first time comment this out
#! ray status

ray.shutdown()
ray.init(num_gpus=1, num_cpus=8, ignore_reinit_error=True)

main(config)
```


#**TODO: Evaluating Policy**

This section crashes the notebook often for some reason, no idea why...


```python
def load_model():
    from torch import load
    path = F"/content/drive/MyDrive/RL_Final_Project/controller.th"
    m = Network(16)
    m.load_state_dict(torch.load(path))
    m.eval()
    return m

def eval():
  # reworking...
  rollout = Rollout('lighthouse')
  rollout.rollout_eval(DiscretizedPolicy(load_model(), 16, 0.1), max_step=100000, frame_skip=1, gamma=0.9)

'''
eval()
'''
```




    '\neval()\n'


