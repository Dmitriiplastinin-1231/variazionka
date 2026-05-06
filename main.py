import tkinter as tk
from tkinter import font as tkfont
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
        self.root.geometry("1120x760")

        self._init_state()
        self._configure_styles()
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

    def _configure_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        base_font = tkfont.nametofont("TkDefaultFont")
        title_font = base_font.copy()
        title_font.configure(size=12, weight="bold")
        section_font = base_font.copy()
        section_font.configure(weight="bold")

        style.configure("Title.TLabel", font=title_font)
        style.configure("Section.TLabel", font=section_font)
        style.configure("Card.TLabelframe", padding=8)
        style.configure("Card.TLabelframe.Label", font=section_font)
        style.configure("Accent.TButton", padding=(10, 4))

    def _build_plot_area(self):
        self.fig = Figure(figsize=(6.4, 5.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#fcfcfc")

        plot_frame = ttk.Frame(self.root, padding=(8, 10))
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        header = ttk.Label(plot_frame, text="График траекторий", style="Title.TLabel")
        header.pack(anchor="w", pady=(0, 6))

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X, pady=(6, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.canvas.draw()

    def _build_controls(self):
        control_frame = ttk.Frame(self.root, padding=(12, 12))
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        title = ttk.Label(control_frame, text="Панель оптимизации", style="Title.TLabel")
        title.pack(anchor="w")
        subtitle = ttk.Label(
            control_frame,
            text="Настройте функцию, методы и параметры запуска.",
        )
        subtitle.pack(anchor="w", pady=(0, 10))

        func_frame = ttk.Labelframe(control_frame, text="Целевая функция", style="Card.TLabelframe")
        func_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(func_frame, text="Формулы:", style="Section.TLabel").pack(
            anchor="w", pady=(0, 2)
        )
        self.f1_label = ttk.Label(
            func_frame, text="f_1(x) = 100(x_2 - x_1^2)^2 + 5(1 - x_1)^2"
        )
        self.f1_label.pack(anchor="w")
        self.f2_label = ttk.Label(
            func_frame,
            text="f_2(x) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2",
        )
        self.f2_label.pack(anchor="w", pady=(0, 6))

        radio_frame = ttk.Frame(func_frame)
        radio_frame.pack(anchor="w")
        ttk.Radiobutton(
            radio_frame, text="f1", variable=self.func_var, value="f1"
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            radio_frame, text="f2", variable=self.func_var, value="f2"
        ).pack(side=tk.LEFT)

        methods_frame = ttk.Labelframe(
            control_frame, text="Методы", style="Card.TLabelframe"
        )
        methods_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(methods_frame, text="Выберите несколько:", style="Section.TLabel").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Checkbutton(
            methods_frame, text="Нелдера-Мида", variable=self.methods["nelder"]
        ).pack(anchor="w")
        ttk.Checkbutton(
            methods_frame, text="Градиентный спуск", variable=self.methods["grad"]
        ).pack(anchor="w")
        ttk.Checkbutton(
            methods_frame,
            text="Сопряжённые градиенты",
            variable=self.methods["cg"],
        ).pack(anchor="w")
        ttk.Checkbutton(
            methods_frame,
            text="Классический Ньютон",
            variable=self.methods["newton"],
        ).pack(anchor="w")

        start_frame = ttk.Labelframe(
            control_frame, text="Начальная точка", style="Card.TLabelframe"
        )
        start_frame.pack(fill=tk.X, pady=(0, 10))
        coords_frame = ttk.Frame(start_frame)
        coords_frame.pack(anchor="w")
        ttk.Label(coords_frame, text="x0", width=3).grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(coords_frame, textvariable=self.x0_var, width=8).grid(
            row=0, column=1, padx=(0, 10)
        )
        ttk.Label(coords_frame, text="y0", width=3).grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(coords_frame, textvariable=self.y0_var, width=8).grid(row=0, column=3)

        bounds_frame = ttk.Labelframe(
            control_frame, text="Область графика", style="Card.TLabelframe"
        )
        bounds_frame.pack(fill=tk.X, pady=(0, 10))
        grid_frame = ttk.Frame(bounds_frame)
        grid_frame.pack(anchor="w")
        ttk.Label(grid_frame, text="xmin").grid(row=0, column=0)
        ttk.Label(grid_frame, text="xmax").grid(row=0, column=1)
        ttk.Label(grid_frame, text="ymin").grid(row=0, column=2)
        ttk.Label(grid_frame, text="ymax").grid(row=0, column=3)
        ttk.Entry(grid_frame, textvariable=self.xmin_var, width=6).grid(row=1, column=0)
        ttk.Entry(grid_frame, textvariable=self.xmax_var, width=6).grid(row=1, column=1)
        ttk.Entry(grid_frame, textvariable=self.ymin_var, width=6).grid(row=1, column=2)
        ttk.Entry(grid_frame, textvariable=self.ymax_var, width=6).grid(row=1, column=3)

        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(action_frame, text="Оптимизировать", style="Accent.TButton", command=self.optimize).pack(
            fill=tk.X
        )

        result_frame = ttk.Labelframe(
            control_frame, text="Результаты", style="Card.TLabelframe"
        )
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(result_frame, height=12, width=50, state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.configure(
            background="#f7f7f7",
            relief="flat",
            font=tkfont.nametofont("TkFixedFont"),
        )

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
        self.ax.grid(True, linestyle="--", alpha=0.5)

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
                    result = nelder_mead(
                        f, start_point, max_iter=1000, tol=1e-6, history=True
                    )
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
                results.append(
                    (method, opt_point, opt_val, hist, colors.get(method, "black"))
                )
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
                    xs[0],
                    ys[0],
                    "o",
                    color=color,
                    markersize=6,
                    markeredgecolor="black",
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
