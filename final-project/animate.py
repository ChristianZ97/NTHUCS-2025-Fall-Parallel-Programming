# animate_3d.py
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_3d(csv_file, gif_file, fps=30, max_frames=300, zoom=0.6):
    """
    zoom < 1 會把鏡頭拉近；1 表示剛好包住所有點
    """

    df = pd.read_csv(csv_file)

    # 所有 step，先排好
    steps = np.sort(df["step"].unique())
    n_steps = len(steps)

    # 每個 step 有 N 筆，假設 id 是 0..N-1 連續
    body_ids = np.sort(df["id"].unique())
    N = len(body_ids)

    # 如果 step 太多，抽樣一些 frame 就好（避免 GIF 太大）
    stride = max(1, n_steps // max_frames)
    used_steps = steps[::stride]

    # 建立 step -> index 對照
    step_to_idx = {s: i for i, s in enumerate(steps)}

    # 依 (step, id) 排序，之後可以用 slice 快速取一個 step 的所有粒子
    df = df.sort_values(["step", "id"]).reset_index(drop=True)

    def get_slice_for_step(step):
        idx = step_to_idx[step]
        start = idx * N
        end = start + N
        return df.iloc[start:end]

    # 算整體範圍，讓視窗固定，不要每 frame 抖來抖去
    all_x = df["x"].to_numpy()
    all_y = df["y"].to_numpy()
    all_z = df["z"].to_numpy()
    xmin, xmax = all_x.min(), all_x.max()
    ymin, ymax = all_y.min(), all_y.max()
    zmin, zmax = all_z.min(), all_z.max()

    # 立方體 bounding box
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    cx = 0.5 * (xmax + xmin)
    cy = 0.5 * (ymax + ymin)
    cz = 0.5 * (zmax + zmin)

    # zoom < 1 -> box 變小，看起來比較 zoom in
    half = 0.5 * max_range * zoom

    fig = plt.figure(figsize=(6, 6), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    ax.set_axis_off()
    ax._axis3don = False

    # 設定座標範圍
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])  # 1:1:1 aspect ratio

    # 初始化第一幀
    first = get_slice_for_step(used_steps[0])
    x0 = first["x"].to_numpy()
    y0 = first["y"].to_numpy()
    z0 = first["z"].to_numpy()
    speed0 = np.sqrt(
        first["vx"].to_numpy() ** 2
        + first["vy"].to_numpy() ** 2
        + first["vz"].to_numpy() ** 2
    )

    # 點大小：N 越大越小，並加上下界
    S_MAX = 30
    S_MIN = 1
    BASE = 20000.0
    s = BASE / max(N, 1)
    s = max(S_MIN, min(S_MAX, s))

    scat = ax.scatter(
        x0, y0, z0, s=s, c=speed0, cmap="plasma", alpha=0.9, edgecolors="none"
    )

    def update(frame_idx):
        step = used_steps[frame_idx]
        sub = get_slice_for_step(step)

        x = sub["x"].to_numpy()
        y = sub["y"].to_numpy()
        z = sub["z"].to_numpy()
        speed = np.sqrt(
            sub["vx"].to_numpy() ** 2
            + sub["vy"].to_numpy() ** 2
            + sub["vz"].to_numpy() ** 2
        )

        scat._offsets3d = (x, y, z)
        scat.set_array(speed)
        return (scat,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(used_steps),
        interval=1000 / fps,
        blit=False,
    )

    writer = PillowWriter(fps=fps)
    ani.save(gif_file, writer=writer)
    print(f"Saved 3D animation to {gif_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python animate_3d.py traj.csv output.gif [zoom]")
    else:
        csv = sys.argv[1]
        out = sys.argv[2]
        if len(sys.argv) >= 4:
            z = float(sys.argv[3])
            animate_3d(csv, out, zoom=z)
        else:
            animate_3d(csv, out)
