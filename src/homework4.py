import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import mujoco_viewer
import environment


class CNP(torch.nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(torch.nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
            self.encoder.append(torch.nn.ReLU())
        self.encoder.append(torch.nn.Linear(hidden_size, hidden_size))
        self.encoder = torch.nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(torch.nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(torch.nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(torch.nn.Linear(hidden_size, hidden_size))
            self.query.append(torch.nn.ReLU())
        self.query.append(torch.nn.Linear(hidden_size, 2 * self.d_y))
        self.query = torch.nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.
        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.
        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class Hw5Env(environment.BaseEnv):
    def __init__(self, render_mode="gui") -> None:
        self._render_mode = render_mode
        self.viewer = None
        self._init_position = [0.0, -np.pi/2, np.pi/2, -2.07, 0, 0, 0]
        self._joint_names = [
            "ur5e/shoulder_pan_joint",
            "ur5e/shoulder_lift_joint",
            "ur5e/elbow_joint",
            "ur5e/wrist_1_joint",
            "ur5e/wrist_2_joint",
            "ur5e/wrist_3_joint",
            "ur5e/robotiq_2f85/right_driver_joint"
        ]
        self.reset()
        self._joint_qpos_idxs = [self.model.joint(x).qposadr for x in self._joint_names]
        self._ee_site = "ur5e/robotiq_2f85/gripper_site"

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        obj_pos = [0.5, 0.0, 1.5]
        height = np.random.uniform(0.03, 0.1)
        self.obj_height = height
        environment.create_object(scene, "box", pos=obj_pos, quat=[0, 0, 0, 1],
                                  size=[0.03, 0.03, height], rgba=[0.8, 0.2, 0.2, 1],
                                  name="obj1")
        return scene

    def state(self):
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="frontface")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=0).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return pixels / 255.0

    def high_level_state(self):
        ee_pos = self.data.site(self._ee_site).xpos[1:]
        obj_pos = self.data.body("obj1").xpos[1:]
        return np.concatenate([ee_pos, obj_pos, [self.obj_height]])


def bezier(p, steps=100):
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    curve = np.power(1-t, 3)*p[0] + 3*np.power(1-t, 2)*t*p[1] + 3*(1-t)*np.power(t, 2)*p[2] + np.power(t, 3)*p[3]
    return curve



def train_cnp(
    model,
    dataset,
    epochs=50,
    lr=1e-3,
    n_context_min=5,
    n_context_max=50,
    n_target_min=10,
    n_target_max=100,
    device="cpu"
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(epochs):
        total_loss = 0.0

        for traj, h in dataset:
            traj = np.asarray(traj)
            T = traj.shape[0]

            n_context = np.random.randint(n_context_min, min(n_context_max, T) + 1)
            n_target = np.random.randint(n_target_min, min(n_target_max, T) + 1)

            idx = np.random.permutation(T)
            ctx_idx = idx[:n_context]
            tgt_idx = idx[:n_target]

            ctx = traj[ctx_idx]
            tgt = traj[tgt_idx]

            t_ctx = torch.tensor(ctx[:, 0:1], dtype=torch.float32)
            y_ctx = torch.tensor(ctx[:, 1:5], dtype=torch.float32)
            h_ctx = torch.full((n_context, 1), h, dtype=torch.float32)

            obs = torch.cat([t_ctx, h_ctx, y_ctx], dim=-1).unsqueeze(0)

            t_tgt = torch.tensor(tgt[:, 0:1], dtype=torch.float32)
            h_tgt = torch.full((n_target, 1), h, dtype=torch.float32)

            target = torch.cat([t_tgt, h_tgt], dim=-1).unsqueeze(0)

            target_truth = torch.tensor(tgt[:, 1:5], dtype=torch.float32).unsqueeze(0)

            loss = model.nll_loss(
                obs.to(device),
                target.to(device),
                target_truth.to(device)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataset):.6f}")

    return model


if __name__ == "__main__":
    env = Hw5Env(render_mode="offscreen") # I don't wanna visualise it for 100 times.
    states_arr = []
    for i in range(100):
        env.reset()
        h = env.obj_height   # the height of the object, which is random and provided by the environment
        p_1 = np.array([0.5, 0.3, 1.04])
        p_2 = np.array([0.5, 0.15, np.random.uniform(1.04, 1.4)])
        p_3 = np.array([0.5, -0.15, np.random.uniform(1.04, 1.4)])
        p_4 = np.array([0.5, -0.3, 1.04])
        points = np.stack([p_1, p_2, p_3, p_4], axis=0)
        curve = bezier(points)

        env._set_ee_in_cartesian(curve[0], rotation=[-90, 0, 180], n_splits=100, max_iters=100, threshold=0.05)
        states = []
        for p in curve:
            env._set_ee_pose(p, rotation=[-90, 0, 180], max_iters=10)
            states.append(env.high_level_state())
        states = np.stack(states)
        states_arr.append((states, h))
        print(f"Collected {i+1} trajectories.", end="\r")
        
    model = CNP(in_shape=(2, 4), hidden_size=128, num_hidden_layers=3)  
    train_cnp(model, states_arr)    
    model.eval()

    n_tests = 100
    ee_mse_list = []
    obj_mse_list = []
    
    for _ in range(n_tests):
        # sample a random trajectory from dataset
        traj, h = states_arr[np.random.randint(len(states_arr))]
        traj = np.asarray(traj)
        T = traj.shape[0]

        # random context / target split
        n_context = np.random.randint(1, T)
        n_target = np.random.randint(1, T)

        idx = np.random.permutation(T)
        ctx_idx = idx[:n_context]
        tgt_idx = idx[:n_target]

        ctx = traj[ctx_idx]
        tgt = traj[tgt_idx]

        # build context
        t_ctx = torch.tensor(ctx[:, 0:1], dtype=torch.float32)
        y_ctx = torch.tensor(ctx[:, 1:5], dtype=torch.float32)
        h_ctx = torch.full((n_context, 1), h, dtype=torch.float32)
        obs = torch.cat([t_ctx, h_ctx,y_ctx], dim=-1).unsqueeze(0)

        # build query
        t_tgt = torch.tensor(tgt[:, 0:1], dtype=torch.float32)
        h_tgt = torch.full((n_target, 1), h, dtype=torch.float32)
        target = torch.cat([t_tgt, h_tgt], dim=-1).unsqueeze(0)

        target_truth = torch.tensor(tgt[:, 1:5], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred_mean, _ = model(obs, target)

        pred = pred_mean.squeeze(0).numpy()
        gt = target_truth.squeeze(0).numpy()

        # split errors
        ee_mse = np.mean((pred[:, 0:2] - gt[:, 0:2])**2)  # (ey, ez)
        obj_mse = np.mean((pred[:, 2:4] - gt[:, 2:4])**2)  # (oy, oz)

        ee_mse_list.append(ee_mse)
        obj_mse_list.append(obj_mse)

    # stats
    ee_mean, ee_std = np.mean(ee_mse_list), np.std(ee_mse_list)
    obj_mean, obj_std = np.mean(obj_mse_list), np.std(obj_mse_list)

    # bar plot
    plt.figure()
    plt.bar([0, 1], [ee_mean, obj_mean], yerr=[ee_std, obj_std])
    plt.xticks([0, 1], ["End-Effector", "Object"])
    plt.ylabel("MSE")
    plt.title("Prediction Error (mean ± std)")
    plt.show()




fig, ax = plt.subplots(1, 2)

for states, _ in states_arr:
    ax[0].plot(states[:, 0], states[:, 1], alpha=0.2, color="b")
    ax[1].plot(states[:, 2], states[:, 3], alpha=0.2, color="r")

ax[0].set(xlabel="e_y", ylabel="e_z")
ax[1].set(xlabel="o_y", ylabel="o_z")

# force same scale
#ax[0].set_xlim(-0.5, 0.5)
#ax[1].set_xlim(-0.5, 0.5)

#ax[0].set_ylim(-0.5, 0.5)
#ax[1].set_ylim(-0.5, 0.5)
#forcing limits gave me an empty plot, so I commented it out.

plt.show()
