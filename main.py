import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from conjugate_gradient import conjugate_gradient_fr
from functions import f1, f2, grad_f1, grad_f2, hessian_f1, hessian_f2
from gradient_descent import gradient_descent
from nelder_mead import nelder_mead
from newton import newton_method
from vector import Vector


class OptimizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Многомерная безусловная минимизация")
        self.root.geometry("1100x750")

        self._init_state()
        self._build_controls()
        self._build_plot_area()

    def _init_state(self):
        self.func_var = tk.StringVar(value="f1")
        self.methods = {
            "nelder": tk.BooleanVar(value=False),
            "grad": tk.BooleanVar(value=False),
            "cg": tk.BooleanVar(value=False),
            "newton": tk.BooleanVar(value=False),
        }
        self.x0_var = tk.StringVar(value="-1.2")
        self.y0_var = tk.StringVar(value="1.0")
        self.xmin_var = tk.StringVar(value="-2.0")
        self.xmax_var = tk.StringVar(value="2.0")
        self.ymin_var = tk.StringVar(value="-1.0")
        self.ymax_var = tk.StringVar(value="3.0")

    def _build_plot_area(self):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.grid(True)
        self.canvas.draw()

    def _build_controls(self):
        control_frame = ttk.LabelFrame(self.root, text="Параметры", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="Целевая функция:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        self.f1_label = ttk.Label(
            control_frame,
            text="f_1(x) = 100(x_2 - x_1^2)^2 + 5(1 - x_1)^2",
        )
        self.f1_label.grid(row=1, column=0, sticky=tk.W)
        self.f2_label = ttk.Label(
            control_frame,
            text="f_2(x) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2",
        )
        self.f2_label.grid(row=2, column=0, sticky=tk.W)

        ttk.Radiobutton(
            control_frame, text="f1", variable=self.func_var, value="f1"
        ).grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Radiobutton(
            control_frame, text="f2", variable=self.func_var, value="f2"
        ).grid(row=4, column=0, sticky=tk.W)

        ttk.Label(control_frame, text="Методы (выберите несколько):").grid(
            row=5, column=0, sticky=tk.W, pady=(10, 2)
        )
        ttk.Checkbutton(
            control_frame, text="Нелдера-Мида", variable=self.methods["nelder"]
        ).grid(row=6, column=0, sticky=tk.W)
        ttk.Checkbutton(
            control_frame, text="Градиентный спуск", variable=self.methods["grad"]
        ).grid(row=7, column=0, sticky=tk.W)
        ttk.Checkbutton(
            control_frame,
            text="Сопряжённые градиенты",
            variable=self.methods["cg"],
        ).grid(row=8, column=0, sticky=tk.W)
        ttk.Checkbutton(
            control_frame,
            text="Классический Ньютон",
            variable=self.methods["newton"],
        ).grid(row=9, column=0, sticky=tk.W)

        ttk.Label(control_frame, text="Начальная точка:").grid(
            row=10, column=0, sticky=tk.W, pady=(10, 2)
        )
        frame_xy = ttk.Frame(control_frame)
        frame_xy.grid(row=11, column=0, sticky=tk.W)
        ttk.Label(frame_xy, text="x0 =").pack(side=tk.LEFT)
        ttk.Entry(frame_xy, textvariable=self.x0_var, width=8).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Label(frame_xy, text="y0 =").pack(side=tk.LEFT)
        ttk.Entry(frame_xy, textvariable=self.y0_var, width=8).pack(side=tk.LEFT)

        ttk.Label(control_frame, text="Область графика:").grid(
            row=12, column=0, sticky=tk.W, pady=(10, 2)
        )
        frame_bounds = ttk.Frame(control_frame)
        frame_bounds.grid(row=13, column=0, sticky=tk.W)
        ttk.Label(frame_bounds, text="xmin").grid(row=0, column=0)
        ttk.Entry(frame_bounds, textvariable=self.xmin_var, width=6).grid(
            row=1, column=0
        )
        ttk.Label(frame_bounds, text="xmax").grid(row=0, column=1)
        ttk.Entry(frame_bounds, textvariable=self.xmax_var, width=6).grid(
            row=1, column=1
        )
        ttk.Label(frame_bounds, text="ymin").grid(row=0, column=2)
        ttk.Entry(frame_bounds, textvariable=self.ymin_var, width=6).grid(
            row=1, column=2
        )
        ttk.Label(frame_bounds, text="ymax").grid(row=0, column=3)
        ttk.Entry(frame_bounds, textvariable=self.ymax_var, width=6).grid(
            row=1, column=3
        )

        ttk.Button(control_frame, text="Оптимизировать", command=self.optimize).grid(
            row=14, column=0, pady=15
        )

        self.result_text = tk.Text(control_frame, height=12, width=50, state=tk.DISABLED)
        self.result_text.grid(row=15, column=0, pady=10)

    def _read_float(self, var, error_message):
        try:
            return float(var.get())
        except ValueError:
            messagebox.showerror("Ошибка", error_message)
            raise

    def get_function_data(self):
        if self.func_var.get() == "f1":
            return f1, grad_f1, hessian_f1, (-2, 2, -1, 3)
        else:
            return f2, grad_f2, hessian_f2, (-5, 5, -5, 5)

    def draw_contours(self, f, xmin, xmax, ymin, ymax):
        """Рисует контуры на текущей оси."""
        x = np.linspace(xmin, xmax, 200)
        y = np.linspace(ymin, ymax, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = f(Vector(X[i, j], Y[i, j]))

        self.ax.clear()
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        func_name = "f1" if self.func_var.get() == "f1" else "f2"
        self.ax.set_title(f"Контуры функции {func_name}")
        self.ax.grid(True)

    def optimize(self):
        selected = [key for key, flag in self.methods.items() if flag.get()]
        if not selected:
            messagebox.showerror("Ошибка", "Выберите хотя бы один метод")
            return

        try:
            x0 = self._read_float(self.x0_var, "Неверные числовые значения")
            y0 = self._read_float(self.y0_var, "Неверные числовые значения")
            xmin = self._read_float(self.xmin_var, "Неверные числовые значения")
            xmax = self._read_float(self.xmax_var, "Неверные числовые значения")
            ymin = self._read_float(self.ymin_var, "Неверные числовые значения")
            ymax = self._read_float(self.ymax_var, "Неверные числовые значения")
        except ValueError:
            return

        start_point = Vector(x0, y0)
        f, grad, hessian, default_bounds = self.get_function_data()

        self.draw_contours(f, xmin, xmax, ymin, ymax)

        colors = {
            "nelder": "red",
            "grad": "blue",
            "cg": "green",
            "newton": "orange",
        }

        results = []
        for method in selected:
            try:
                if method == "nelder":
                    result = nelder_mead(f, start_point, max_iter=1000, tol=1e-6, history=True)
                elif method == "grad":
                    result = gradient_descent(
                        f, grad, start_point, max_iter=1000, tol=1e-6, history=True
                    )
                elif method == "cg":
                    result = conjugate_gradient_fr(
                        f, grad, start_point, max_iter=1000, tol=1e-6, history=True
                    )
                elif method == "newton":
                    result = newton_method(
                        f,
                        grad,
                        hessian,
                        start_point,
                        max_iter=1000,
                        tol=1e-6,
                        use_line_search=True,
                        history=True,
                    )
                else:
                    continue

                opt_point, opt_val, hist = result
                results.append((method, opt_point, opt_val, hist, colors.get(method, "black")))
            except Exception as exc:
                messagebox.showerror(
                    "Ошибка", f"Метод {method} завершился ошибкой: {str(exc)}"
                )

        if not results:
            messagebox.showerror("Ошибка", "Ни один метод не выполнен успешно")
            return

        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        for method, opt_point, opt_val, hist, _ in results:
            self.result_text.insert(tk.END, f"Метод {method}:\n")
            self.result_text.insert(tk.END, f"  Минимум: {opt_point}\n")
            self.result_text.insert(tk.END, f"  Значение: {opt_val:.6f}\n")
            iterations = len(hist) - 1 if hist else 0
            self.result_text.insert(tk.END, f"  Итераций: {iterations}\n\n")
        self.result_text.config(state=tk.DISABLED)

        for method, opt_point, opt_val, hist, color in results:
            if hist:
                xs = [p.x for p in hist]
                ys = [p.y for p in hist]
                self.ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=color,
                    markersize=3,
                    linewidth=1,
                    label=method,
                    alpha=0.7,
                )
                self.ax.plot(
                    xs[0], ys[0], "o", color=color, markersize=6, markeredgecolor="black"
                )
                self.ax.plot(
                    xs[-1],
                    ys[-1],
                    "s",
                    color=color,
                    markersize=6,
                    markeredgecolor="black",
                )
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizationGUI(root)
    root.mainloop()
