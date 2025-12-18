
from manim import *

class DerivativeVisualization(Scene):
    def construct(self):
        # 1. 创建坐标系
        axes = Axes(
            x_range=[-1, 5], y_range=[-1, 10],
            axis_config={"color": BLUE}
        )
        
        # 2. 定义函数 f(x) = x^2
        func = axes.plot(lambda x: 0.5 * x**2, color=YELLOW)
        func_label = MathTex("f(x) = 0.5x^2").next_to(func, UP)

        # 3. 定义两个点：x (固定点) 和 x + dx (移动点)
        x_val = 2
        dx_tracker = ValueTracker(2.0)  # 初始 dx = 2.0，也就是两点离很远

        # 4. 动态创建切线/割线
        # always_redraw 是 Manim 的精髓：每帧都会重新运行这个函数
        def get_secant_line():
            dx = dx_tracker.get_value()
            x1 = x_val
            x2 = x_val + dx
            y1 = 0.5 * x1**2
            y2 = 0.5 * x2**2
            
            # 计算斜率 (f(x+dx) - f(x)) / dx
            slope = (y2 - y1) / (x2 - x1) if dx != 0 else 2.0 # 防止除以0
            
            # 画线
            return axes.get_secant_slope_group(
                x=x_val,
                graph=func,
                dx=dx,
                dx_line_color=TEAL,
                df_line_color=RED, # 这里的斜率线就是导数的可视化
                secant_line_color=GREEN,
            )

        secant_group = always_redraw(get_secant_line)

        # 5. 添加到场景
        self.add(axes, func, func_label, secant_group)
        self.wait(1)

        # 6. 动画：让 dx 从 2 变成 0.01 (模拟求极限的过程)
        # 这就是“求导”的本质：dx -> 0
        self.play(dx_tracker.animate.set_value(0.01), run_time=4)
        self.wait()
        
        
        