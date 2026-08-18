"""
PID 控制器有效性验证脚本
========================
模拟对比：无控制（开环）vs P-only vs PI vs PID 对二阶系统的控制效果。
评估指标：上升时间、调节时间、超调量、稳态误差。

用法：
    python pid_verify.py

依赖：numpy, matplotlib (均为标准科学计算库)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys


# ============================================================
# 1. 系统模型：二阶传递函数  G(s) = K / (tau^2*s^2 + 2*zeta*tau*s + 1)
# ============================================================
class SecondOrderPlant:
    """二阶系统 + 输出限幅 + 测量噪声"""
    def __init__(self, K=1.0, tau=0.5, zeta=0.3, noise_std=0.0, y_max=5.0):
        self.K = K
        self.tau = tau
        self.zeta = zeta
        self.noise_std = noise_std
        self.y_max = y_max
        self.reset()

    def reset(self):
        self.y = 0.0
        self.dy = 0.0

    def step(self, u, dt):
        ddy = (self.K * u - self.y - 2 * self.zeta * self.tau * self.dy) / (self.tau ** 2)
        self.dy += ddy * dt
        self.y += self.dy * dt
        self.y = np.clip(self.y, -self.y_max, self.y_max)
        noise = np.random.normal(0, self.noise_std) if self.noise_std > 0 else 0.0
        return self.y + noise


# ============================================================
# 2. PID 控制器（含积分抗饱和、微分低通滤波）
# ============================================================
class PIDController:
    def __init__(self, Kp=0.0, Ki=0.0, Kd=0.0, setpoint=1.0,
                 output_min=-10.0, output_max=10.0, tau_d=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self.tau_d = tau_d
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_deriv = 0.0

    def update(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        raw_deriv = (error - self.prev_error) / dt if dt > 0 else 0.0
        alpha = dt / (self.tau_d + dt) if (self.tau_d + dt) > 0 else 1.0
        self.filtered_deriv = (1 - alpha) * self.filtered_deriv + alpha * raw_deriv
        self.prev_error = error

        u = (self.Kp * error + self.Ki * self.integral + self.Kd * self.filtered_deriv)
        u_sat = np.clip(u, self.output_min, self.output_max)
        if u != u_sat:
            self.integral -= (u - u_sat) * dt * 0.5
        return u_sat


# ============================================================
# 3. 仿真主循环
# ============================================================
def simulate(plant, controller, t_end=15.0, dt=0.01):
    n_steps = int(t_end / dt)
    t_arr = np.linspace(0, t_end, n_steps)
    y_arr = np.zeros(n_steps)
    u_arr = np.zeros(n_steps)
    sp_arr = np.full(n_steps, controller.setpoint)
    plant.reset()
    controller.reset()

    for i in range(n_steps):
        if i == int(n_steps * 0.6):
            controller.setpoint = 1.5  # 中途改变设定值
        u_prev = u_arr[i-1] if i > 0 else 0.0
        y_arr[i] = plant.step(u_prev, dt)
        u_arr[i] = controller.update(y_arr[i], dt)

    return t_arr, y_arr, u_arr, sp_arr


# ============================================================
# 4. 评价指标计算
# ============================================================
def compute_metrics(t, y, sp, tol=0.02):
    final_sp = sp[-1]
    e_ss = abs(final_sp - y[-1])

    # 上升时间 (10% -> 90%)
    y10, y90 = 0.1 * final_sp, 0.9 * final_sp
    t_rise, t_rise_start = None, None
    for i in range(len(t)):
        if t_rise_start is None and y[i] >= y10:
            t_rise_start = t[i]
        if t_rise_start is not None and y[i] >= y90:
            t_rise = t[i] - t_rise_start
            break

    # 超调量
    overshoot = max(0.0, (np.max(y) - final_sp) / final_sp * 100)

    # 调节时间 (+/- tol band)
    band = tol * abs(final_sp)
    t_settle = None
    for i in range(len(t)-1, -1, -1):
        if abs(y[i] - final_sp) > band:
            t_settle = t[min(i+1, len(t)-1)]
            break

    # ISE (积分平方误差)
    ise = np.trapz((y - sp[:len(y)])**2, t[:len(y)])

    return {
        "Steady Error": e_ss,
        "Rise Time [s]": t_rise,
        "Overshoot [%]": overshoot,
        "Settling Time [s]": t_settle,
        "ISE": ise,
    }


# ============================================================
# 5. 简易手动调参（Ziegler-Nichols 启发式）
# ============================================================
def auto_tune(plant, dt):
    # 粗略扫描找到临界增益 Ku
    Ku, Tu = None, None
    for Kp_test in np.linspace(0.5, 15, 60):
        plant.reset()
        ctrl = PIDController(Kp=Kp_test, Ki=0.0, Kd=0.0, setpoint=1.0)
        y_vals = []
        for _ in range(300):
            y = plant.step(ctrl.update(y, dt), dt)
            y_vals.append(y)
        last = np.array(y_vals[-60:])
        peaks = 0
        for j in range(2, len(last)-2):
            if last[j] > last[j-1] and last[j] > last[j-2] and \
               last[j] > last[j+1] and last[j] > last[j+2]:
                peaks += 1
        if peaks >= 4:
            Ku = Kp_test
            Tu = dt * 60 / peaks
            break
    if Ku is None:
        Ku, Tu = 3.0, 1.2

    return {
        "P":   (0.5 * Ku, 0.0, 0.0),
        "PI":  (0.45 * Ku, 0.54 * Ku / Tu, 0.0),
        "PID": (0.6 * Ku, 1.2 * Ku / Tu, 0.075 * Ku * Tu),
    }


# ============================================================
# 6. 主程序
# ============================================================
def main():
    print("=" * 60)
    print("PID CONTROLLER VERIFICATION")
    print("=" * 60)

    dt = 0.01
    t_end = 15.0
    plant = SecondOrderPlant(K=1.0, tau=0.5, zeta=0.3, noise_std=0.005)

    # 自动调参
    gains = auto_tune(plant, dt)
    print("\nAuto-tune reference (Ziegler-Nichols):")
    for name, (kp, ki, kd) in gains.items():
        print(f"  {name:>5s}: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")

    # 手动微调后的控制器
    controllers = {
        "Open-Loop": PIDController(Kp=0.0, Ki=0.0, Kd=0.0, setpoint=1.0),
        "P-only":    PIDController(Kp=2.0, Ki=0.0, Kd=0.0, setpoint=1.0),
        "PI":        PIDController(Kp=1.6, Ki=2.5, Kd=0.0, setpoint=1.0),
        "PID":       PIDController(Kp=1.8, Ki=3.0, Kd=0.12, setpoint=1.0),
    }

    # 仿真
    results = {}
    for name, ctrl in controllers.items():
        t, y, u, sp = simulate(plant, ctrl, t_end, dt)
        metrics = compute_metrics(t, y, sp)
        results[name] = (t, y, u, sp, metrics)

    # 打印指标表
    header = f"{'Controller':<14s} {'Steady Err':>10s} {'Rise(s)':>10s} {'Overshoot%':>11s} {'Settle(s)':>10s} {'ISE':>10s}"
    print("\n" + header)
    print("-" * len(header))
    for name, (_, y, _, sp, m) in results.items():
        rise = f"{m['Rise Time [s]']:.3f}" if m['Rise Time [s]'] else "N/A"
        settle = f"{m['Settling Time [s]']:.3f}" if m['Settling Time [s]'] else "N/A"
        print(f"{name:<14s} {m['Steady Error']:>10.4f} {rise:>10s} {m['Overshoot [%]']:>10.2f} {settle:>10s} {m['ISE']:>10.4f}")

    # 绘图
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    colors = {"Open-Loop": "gray", "P-only": "orange", "PI": "blue", "PID": "red"}
    lstyle = {"Open-Loop": "--", "P-only": "-.", "PI": "-", "PID": "-"}

    # 子图1：系统输出
    ax1 = fig.add_subplot(gs[0, :])
    for name, (t, y, u, sp, _) in results.items():
        ax1.plot(t, y, color=colors[name], ls=lstyle[name], lw=1.8, label=name)
    ax1.plot(t, sp, "k--", lw=1.0, alpha=0.5, label="Setpoint")
    ax1.axvline(x=0.6*t_end, color="gray", ls=":", alpha=0.5, label="SP change")
    ax1.set_xlabel("Time [s]"); ax1.set_ylabel("Output y(t)")
    ax1.set_title("System Output Response"); ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)

    # 子图2：控制输入
    ax2 = fig.add_subplot(gs[1, 0])
    for name, (t, y, u, sp, _) in results.items():
        if name == "Open-Loop": continue
        ax2.plot(t, u, color=colors[name], ls=lstyle[name], lw=1.5, label=name)
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Control Input u(t)")
    ax2.set_title("Control Input"); ax2.legend(); ax2.grid(alpha=0.3)

    # 子图3：跟踪误差
    ax3 = fig.add_subplot(gs[1, 1])
    for name, (t, y, u, sp, _) in results.items():
        if name == "Open-Loop": continue
        e = sp - y
        ax3.plot(t, e, color=colors[name], ls=lstyle[name], lw=1.5, label=name)
    ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Error e(t)")
    ax3.set_title("Tracking Error"); ax3.legend(); ax3.grid(alpha=0.3)

    fig.suptitle("PID Controller Verification - Second-Order Plant", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("pid_verify_results.png", dpi=150)
    print("\n[Chart saved to: pid_verify_results.png]")
    plt.show()

    # 有效性判定
    print("\n" + "="*60)
    print("VALIDATION RESULT:")
    print("="*60)
    pid_m = results["PID"][4]
    checks = [
        ("Steady Error < 5%",    pid_m["Steady Error"] < 0.05 * 1.5),
        ("Overshoot < 20%",      pid_m["Overshoot [%]"] < 20),
        ("Settling Time < 5s",   pid_m["Settling Time [s]"] is not None and pid_m["Settling Time [s]"] < 5.0),
        ("Rise Time exists",     pid_m["Rise Time [s]"] is not None),
    ]
    all_pass = True
    for desc, passed in checks:
        status = "[PASS]" if passed else "[FAIL]"
        if not passed: all_pass = False
        print(f"  {status}  {desc}")

    if all_pass:
        print("\n  *** PID CONTROLLER VERIFICATION PASSED ***")
    else:
        print("\n  *** Some metrics need improvement - adjust parameters ***")

    return 0


if __name__ == "__main__":
    sys.exit(main())
