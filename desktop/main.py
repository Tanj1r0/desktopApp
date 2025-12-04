import tkinter as tk
from tkinter import ttk, messagebox, filedialog, font
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime

# Настройка стиля matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# -------------------------- ЛОГИКА РАСЧЁТА --------------------------
def calculate_forecast(values, alpha):
    """
    Прогнозирование методом экспоненциального сглаживания квадратичного тренда
    """
    y = np.array(values, dtype=float)
    n_obs = len(y)
    t = np.arange(1, n_obs + 1)

    # 1. Квадратичный тренд (полином 2-й степени)
    X = np.vstack([np.ones_like(t), t, t ** 2]).T
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0, a1, a2 = coeffs

    total_steps = 13
    rows = []

    # 2. Начальные S0 по формулам из документа
    s01 = a0 - a1 * (1 - alpha) / alpha + a2 * (1 - alpha) * (2 - alpha) / (2 * alpha ** 2)
    s02 = a0 - 2 * a1 * (1 - alpha) / alpha + a2 * (1 - alpha) * (3 - 2 * alpha) / (alpha ** 2)
    s03 = a0 - 3 * a1 * (1 - alpha) / alpha + 3 * a2 * (1 - alpha) * (4 - 3 * alpha) / (2 * alpha ** 2)

    # 3. Первые экспоненциальные средние
    s1 = alpha * (a0 + a1 * 1 + a2 * 1 ** 2) + (1 - alpha) * s01
    s2 = alpha * s1 + (1 - alpha) * s02
    s3 = alpha * s2 + (1 - alpha) * s03

    # 4. Расчет среднеквадратической ошибки
    trend_obs = a0 + a1 * t + a2 * (t ** 2)
    residuals = trend_obs - y
    disp = np.sum((residuals - residuals.mean()) ** 2) / (len(residuals) - 1)
    kvadr = sqrt(disp)

    # 5. Прогноз на 13 периодов
    for j in range(1, total_steps + 1):
        # Коэффициенты для прогноза
        a0_qua = 3 * (s1 - s2) + s3
        remp = (6 - 5 * alpha) * s1 - 2 * (5 - 4 * alpha) * s2 + (4 - 3 * alpha) * s3
        a1_qua = remp * alpha / (2 * (1 - alpha) ** 2)
        a2_qua = (s1 - 2 * s2 + s3) * alpha ** 2 / ((1 - alpha) ** 2)

        # Прогнозное значение
        forecast = a0_qua + a1_qua * j + 0.5 * a2_qua * j ** 2

        # Ошибка прогноза
        err = kvadr * sqrt(2 * alpha + 3 * alpha ** 2 + 3 * (alpha ** 3) * (j ** 2))

        # Доверительные интервалы
        upper = forecast + err
        lower = forecast - err

        # Сохраняем все промежуточные результаты
        rows.append([
            2003 + j,  # Год
            round(s1, 4), round(s2, 4), round(s3, 4),
            round(a0_qua, 4), round(a1_qua, 4), round(a2_qua, 4),
            round(forecast, 2), round(err, 2),
            round(upper, 2), round(lower, 2)
        ])

        # Обновление начальных условий для следующей итерации
        s01 = a0_qua - a1_qua * (1 - alpha) / alpha + a2_qua * (1 - alpha) * (2 - alpha) / (2 * alpha ** 2)
        s02 = a0_qua - 2 * a1_qua * (1 - alpha) / alpha + a2_qua * (1 - alpha) * (3 - 2 * alpha) / (alpha ** 2)
        s03 = a0_qua - 3 * a1_qua * (1 - alpha) / alpha + 3 * a2_qua * (1 - alpha) * (4 - 3 * alpha) / (2 * alpha ** 2)

        s1 = alpha * (a0_qua + a1_qua * 1 + 0.5 * a2_qua * 1 ** 2) + (1 - alpha) * s01
        s2 = alpha * s1 + (1 - alpha) * s02
        s3 = alpha * s2 + (1 - alpha) * s03

    # Создаем DataFrame с результатами
    df = pd.DataFrame(rows, columns=[
        "Год", "S1", "S2", "S3", "A0", "A1", "A2",
        "Прогноз", "Ошибка", "Верхняя", "Нижняя"
    ])

    return df, (a0, a1, a2), y


# -------------------------- СТИЛИ И ЦВЕТА --------------------------
class Colors:
    """Цветовая схема приложения"""
    PRIMARY = "#2C3E50"  # Темно-синий
    SECONDARY = "#34495E"  # Светло-синий
    ACCENT = "#3498DB"  # Голубой
    SUCCESS = "#2ECC71"  # Зеленый
    WARNING = "#F39C12"  # Оранжевый
    DANGER = "#E74C3C"  # Красный
    LIGHT = "#ECF0F1"  # Светло-серый
    DARK = "#2C3E50"  # Темный
    WHITE = "#FFFFFF"  # Белый
    GRAY = "#95A5A6"  # Серый

    CHART_COLORS = ["#3498DB", "#2ECC71", "#E74C3C", "#9B59B6", "#F1C40F"]


# -------------------------- КАСТОМНЫЕ ВИДЖЕТЫ --------------------------
class ModernButton(tk.Button):
    """Современная кнопка с градиентом"""

    def __init__(self, master=None, **kwargs):
        bg_color = kwargs.pop('bg_color', Colors.ACCENT)
        fg_color = kwargs.pop('fg_color', Colors.WHITE)
        hover_color = kwargs.pop('hover_color', "#2980B9")

        super().__init__(master, **kwargs)

        self.config(
            font=("Segoe UI", 10, "bold"),
            bg=bg_color,
            fg=fg_color,
            activebackground=hover_color,
            activeforeground=fg_color,
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor="hand2"
        )

        self.hover_color = hover_color
        self.default_color = bg_color

        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, e):
        self.config(bg=self.hover_color)

    def on_leave(self, e):
        self.config(bg=self.default_color)


class CardFrame(tk.Frame):
    """Карточка с тенью"""

    def __init__(self, master=None, title="", **kwargs):
        bg_color = kwargs.pop('bg', Colors.WHITE)
        super().__init__(master, bg=bg_color, **kwargs)

        self.config(
            highlightbackground=Colors.GRAY,
            highlightthickness=1,
            relief=tk.RAISED,
            bd=0
        )

        if title:
            title_label = tk.Label(
                self,
                text=title,
                font=("Segoe UI", 11, "bold"),
                bg=bg_color,
                fg=Colors.PRIMARY,
                anchor="w"
            )
            title_label.pack(fill=tk.X, padx=15, pady=(10, 5))

            # Разделитель
            separator = tk.Frame(self, height=2, bg=Colors.ACCENT)
            separator.pack(fill=tk.X, padx=15, pady=(0, 10))


class ModernEntry(tk.Entry):
    """Современное поле ввода"""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.config(
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            bd=2,
            highlightbackground=Colors.GRAY,
            highlightcolor=Colors.ACCENT,
            highlightthickness=1
        )


class ModernText(tk.Text):
    """Современное текстовое поле"""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.config(
            font=("Segoe UI", 10),
            relief=tk.FLAT,
            bd=2,
            highlightbackground=Colors.GRAY,
            highlightcolor=Colors.ACCENT,
            highlightthickness=1,
            wrap=tk.WORD
        )


# -------------------------- ГЛАВНОЕ ПРИЛОЖЕНИЕ --------------------------
class ForecastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 Прогнозирование - Метод экспоненциального сглаживания")
        self.root.geometry("1400x800")

        # Центрирование окна
        self.center_window()

        # Иконка (если есть)
        try:
            self.root.iconbitmap('chart.ico')
        except:
            pass

        # Настройка стилей
        self.setup_styles()

        # Данные
        self.df = None
        self.y = None
        self.trend_coeffs = None

        # Создание интерфейса
        self.create_widgets()

        # Загрузить пример данных
        self.load_example_data()

    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'1400x800+{x}+{y}')

    def setup_styles(self):
        """Настройка стилей приложения"""
        self.root.configure(bg=Colors.LIGHT)

        # Создаем кастомные стили
        style = ttk.Style()
        style.theme_use('clam')

        # Настройка вкладок
        style.configure(
            "Custom.TNotebook",
            background=Colors.LIGHT,
            borderwidth=0
        )

        style.configure(
            "Custom.TNotebook.Tab",
            background=Colors.GRAY,
            foreground=Colors.WHITE,
            padding=[20, 10],
            font=("Segoe UI", 10, "bold")
        )

        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", Colors.ACCENT)],
            foreground=[("selected", Colors.WHITE)]
        )

    def create_widgets(self):
        """Создание интерфейса"""
        # Верхняя панель с заголовком
        self.create_header()

        # Основной контейнер
        main_container = tk.Frame(self.root, bg=Colors.LIGHT)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Левая панель (ввод данных)
        left_panel = tk.Frame(main_container, bg=Colors.LIGHT)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        # Правая панель (результаты)
        right_panel = tk.Frame(main_container, bg=Colors.LIGHT)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Создание левой панели
        self.create_left_panel(left_panel)

        # Создание правой панели
        self.create_right_panel(right_panel)

    def create_header(self):
        """Создание верхней панели с заголовком"""
        header = tk.Frame(self.root, bg=Colors.PRIMARY, height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Заголовок
        title_label = tk.Label(
            header,
            text="📊 ПРОГНОЗИРОВАНИЕ МЕТОДОМ ЭКСПОНЕНЦИАЛЬНОГО СГЛАЖИВАНИЯ",
            font=("Segoe UI", 16, "bold"),
            bg=Colors.PRIMARY,
            fg=Colors.WHITE
        )
        title_label.pack(side=tk.LEFT, padx=30, pady=20)

        # Информация о версии
        version_label = tk.Label(
            header,
            text="Версия 1.0",
            font=("Segoe UI", 9),
            bg=Colors.PRIMARY,
            fg=Colors.GRAY
        )
        version_label.pack(side=tk.RIGHT, padx=30, pady=20)

    def create_left_panel(self, parent):
        """Создание левой панели с полной вертикальной прокруткой"""

        # ---- 1. Создаем Canvas + Scrollbar ----
        canvas = tk.Canvas(parent, bg=Colors.LIGHT, highlightthickness=0, width=420)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # ---- 2. Внутренний Frame (именно он будет прокручиваться) ----
        scroll_frame = tk.Frame(canvas, bg=Colors.LIGHT, width=410)
        scroll_frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        # ---- Автоматическое изменение размеров ----
        def configure_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            # Чтобы ширина scroll_frame совпадала с шириной canvas
            canvas.itemconfig(scroll_frame_id, width=event.width)

        scroll_frame.bind("<Configure>", configure_frame)

        # Поддержка прокрутки колесом мыши
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # ---- 3. Далее помещаем ВСЕ карточки внутрь scroll_frame ----

        # -------------------- Карточка ввода --------------------
        input_card = CardFrame(scroll_frame, title="ВВОД ДАННЫХ", bg=Colors.WHITE, width=400)
        input_card.pack(fill=tk.X, pady=(0, 15))

        content_frame = tk.Frame(input_card, bg=Colors.WHITE, padx=15, pady=15)
        content_frame.pack(fill=tk.X)

        # Поле ввода значений
        tk.Label(
            content_frame,
            text="Введите 10 значений (через запятую):",
            font=("Segoe UI", 10, "bold"),
            bg=Colors.WHITE,
            fg=Colors.PRIMARY
        ).pack(anchor="w", pady=(0, 5))

        self.values_text = ModernText(content_frame, height=6, width=40)
        self.values_text.pack(fill=tk.X, pady=(0, 15))

        scrollbar_values = tk.Scrollbar(content_frame, command=self.values_text.yview)
        scrollbar_values.pack(side=tk.RIGHT, fill=tk.Y)
        self.values_text.config(yscrollcommand=scrollbar_values.set)

        # Поле α
        tk.Label(
            content_frame,
            text="Параметр сглаживания (α):",
            font=("Segoe UI", 10, "bold"),
            bg=Colors.WHITE,
            fg=Colors.PRIMARY
        ).pack(anchor="w", pady=(10, 5))

        alpha_frame = tk.Frame(content_frame, bg=Colors.WHITE)
        alpha_frame.pack(fill=tk.X)

        self.alpha_entry = ModernEntry(alpha_frame, width=20)
        self.alpha_entry.pack(side=tk.LEFT)

        tk.Label(
            alpha_frame,
            text="(0 < α < 1)",
            font=("Segoe UI", 9),
            bg=Colors.WHITE,
            fg=Colors.GRAY
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Кнопки
        button_frame = tk.Frame(content_frame, bg=Colors.WHITE)
        button_frame.pack(fill=tk.X, pady=15)

        ModernButton(
            button_frame,
            text="🚀 Рассчитать",
            bg_color=Colors.SUCCESS,
            hover_color="#27AE60",
            command=self.calculate
        ).pack(side=tk.LEFT, padx=(0, 10))

        ModernButton(
            button_frame,
            text="🔄 Очистить",
            bg_color=Colors.WARNING,
            hover_color="#D68910",
            command=self.clear_data
        ).pack(side=tk.LEFT)

        # -------------------- Примеры --------------------
        examples_card = CardFrame(scroll_frame, title="ПРИМЕРЫ ДАННЫХ", bg=Colors.WHITE, width=400)
        examples_card.pack(fill=tk.X, pady=(0, 15))

        examples_content = tk.Frame(examples_card, bg=Colors.WHITE, padx=15, pady=15)
        examples_content.pack(fill=tk.BOTH)

        examples_btn_frame = tk.Frame(examples_content, bg=Colors.WHITE)
        examples_btn_frame.pack(fill=tk.X, pady=(0, 10))

        ModernButton(
            examples_btn_frame,
            text="📊 Смертность от БСК",
            bg_color=Colors.ACCENT,
            hover_color="#2980B9",
            command=self.load_mortality_example,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT, padx=(0, 10))

        ModernButton(
            examples_btn_frame,
            text="🏥 Заболеваемость",
            bg_color="#9B59B6",
            hover_color="#8E44AD",
            command=self.load_morbidity_example,
            font=("Segoe UI", 9)
        ).pack(side=tk.LEFT)

        tk.Label(
            examples_content,
            text="Смертность (на 10 тыс.):",
            font=("Segoe UI", 9, "bold"),
            bg=Colors.WHITE,
            fg=Colors.DARK
        ).pack(anchor="w")

        tk.Label(
            examples_content,
            text="75.42, 77.87, 70.76, 67.83, 68.59, 67.12, 62.6, 59.32, 61.69, 54.55",
            font=("Consolas", 8),
            bg=Colors.WHITE,
            fg=Colors.GRAY,
            wraplength=300
        ).pack(anchor="w", pady=(0, 10))

        # -------------------- Информация --------------------
        info_card = CardFrame(scroll_frame, title="ИНФОРМАЦИЯ", bg=Colors.WHITE, width=400)
        info_card.pack(fill=tk.X)

        info_content = tk.Frame(info_card, bg=Colors.WHITE, padx=15, pady=15)
        info_content.pack(fill=tk.BOTH)

        info_text = """
    Метод экспоненциального сглаживания:
    • Используется для прогнозирования временных рядов
    • α = 2/(m+1)
    • Рекомендуемые значения: 0.01 - 0.3
    • По умолчанию: α = 0.0625

    Входные данные:
    • 10 значений
    • Дробные числа

    Выходные данные:
    • Прогноз на 13 периодов
    • Доверительные интервалы
    • Визуализация графиков
    """

        tk.Label(
            info_content,
            text=info_text.strip(),
            font=("Segoe UI", 9),
            bg=Colors.WHITE,
            fg=Colors.DARK,
            justify=tk.LEFT,
            wraplength=300
        ).pack(anchor="w")

    def create_right_panel(self, parent):
        """Создание правой панели с результатами"""
        # Создаем вкладки с кастомным стилем
        self.notebook = ttk.Notebook(parent, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Вкладка 1: Таблица результатов
        self.table_frame = CardFrame(self.notebook, bg=Colors.WHITE)
        self.notebook.add(self.table_frame, text="📋 РЕЗУЛЬТАТЫ")

        self.create_table_widget()

        # Вкладка 2: График
        self.chart_frame = CardFrame(self.notebook, bg=Colors.WHITE)
        self.notebook.add(self.chart_frame, text="📈 ГРАФИКИ")

        self.create_chart_widget()

        # Вкладка 3: Статистика
        self.stats_frame = CardFrame(self.notebook, bg=Colors.WHITE)
        self.notebook.add(self.stats_frame, text="📊 СТАТИСТИКА")

        self.create_stats_widget()

        # Панель с кнопками экспорта
        self.create_export_panel(parent)

    def create_table_widget(self):
        """Создание виджета таблицы"""
        # Контейнер для таблицы
        table_container = tk.Frame(self.table_frame, bg=Colors.WHITE)
        table_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Создаем Treeview с красивым стилем
        columns = ("Год", "S1", "S2", "S3", "A0", "A1", "A2", "Прогноз", "Ошибка", "Верхняя", "Нижняя")

        # Стили для Treeview
        style = ttk.Style()
        style.configure(
            "Custom.Treeview",
            background=Colors.WHITE,
            foreground=Colors.DARK,
            rowheight=25,
            fieldbackground=Colors.WHITE,
            font=("Segoe UI", 9)
        )

        style.configure(
            "Custom.Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background=Colors.ACCENT,
            foreground=Colors.WHITE,
            relief=tk.FLAT
        )

        style.map(
            "Custom.Treeview.Heading",
            background=[('active', Colors.SECONDARY)]
        )

        # Создаем Treeview
        self.tree = ttk.Treeview(
            table_container,
            columns=columns,
            show="headings",
            style="Custom.Treeview",
            height=15
        )

        # Настраиваем колонки
        column_widths = [60, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
        for col, width in zip(columns, column_widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=width, anchor="center", minwidth=50)

        # Добавляем прокрутку
        scrollbar_y = ttk.Scrollbar(table_container, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL, command=self.tree.xview)

        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

        # Размещаем виджеты
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        # Настраиваем расширение
        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

    def create_chart_widget(self):
        """Создание виджета графика"""
        # Контейнер для графика
        chart_container = tk.Frame(self.chart_frame, bg=Colors.WHITE)
        chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Создаем фигуру matplotlib
        self.fig, self.ax = plt.subplots(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor(Colors.WHITE)
        self.ax.set_facecolor(Colors.WHITE)

        # Настройка стиля графика
        self.ax.grid(True, alpha=0.3, linestyle='--')

        # Создаем холст
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_container)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Панель управления графиком
        control_frame = tk.Frame(chart_container, bg=Colors.WHITE)
        control_frame.pack(fill=tk.X, pady=(10, 0))

        self.chart_type = tk.StringVar(value="all")

        tk.Radiobutton(
            control_frame,
            text="Все данные",
            variable=self.chart_type,
            value="all",
            font=("Segoe UI", 9),
            bg=Colors.WHITE,
            activebackground=Colors.WHITE,
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            control_frame,
            text="Прогноз",
            variable=self.chart_type,
            value="forecast",
            font=("Segoe UI", 9),
            bg=Colors.WHITE,
            activebackground=Colors.WHITE,
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(
            control_frame,
            text="Тренд",
            variable=self.chart_type,
            value="trend",
            font=("Segoe UI", 9),
            bg=Colors.WHITE,
            activebackground=Colors.WHITE,
            command=self.update_chart
        ).pack(side=tk.LEFT, padx=5)

    def create_stats_widget(self):
        """Создание виджета статистики"""
        # Контейнер для статистики
        stats_container = tk.Frame(self.stats_frame, bg=Colors.WHITE)
        stats_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Создаем текстовое поле с прокруткой
        text_frame = tk.Frame(stats_container, bg=Colors.WHITE)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = tk.Text(
            text_frame,
            font=("Consolas", 10),
            bg="#F8F9FA",
            fg=Colors.DARK,
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )

        scrollbar = tk.Scrollbar(text_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)

        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Заглушка
        self.stats_text.insert(1.0, "Здесь будет отображаться статистика после расчета...")
        self.stats_text.config(state=tk.DISABLED)

    def create_export_panel(self, parent):
        """Создание панели экспорта"""
        export_frame = tk.Frame(parent, bg=Colors.LIGHT, height=60)
        export_frame.pack(fill=tk.X, pady=(15, 0))
        export_frame.pack_propagate(False)

        # Контейнер для кнопок
        button_container = tk.Frame(export_frame, bg=Colors.LIGHT)
        button_container.pack(expand=True)

        ModernButton(
            button_container,
            text="💾 Экспорт в Excel",
            bg_color="#16A085",
            hover_color="#138D75",
            command=self.export_excel,
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            button_container,
            text="🖼️ Сохранить график",
            bg_color="#8E44AD",
            hover_color="#7D3C98",
            command=self.save_chart,
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            button_container,
            text="📋 Копировать таблицу",
            bg_color="#D35400",
            hover_color="#BA4A00",
            command=self.copy_to_clipboard,
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=5)

        ModernButton(
            button_container,
            text="🔄 Обновить",
            bg_color=Colors.GRAY,
            hover_color="#7F8C8D",
            command=self.update_chart,
            font=("Segoe UI", 10)
        ).pack(side=tk.LEFT, padx=5)

    def load_example_data(self):
        """Загрузка примера данных"""
        self.load_mortality_example()

    def load_mortality_example(self):
        """Загрузка примера смертности"""
        example_values = "75.42, 77.87, 70.76, 67.83, 68.59, 67.12, 62.6, 59.32, 61.69, 54.55"

        self.values_text.delete(1.0, tk.END)
        self.values_text.insert(1.0, example_values)
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, "0.0625")

    def load_morbidity_example(self):
        """Загрузка примера заболеваемости"""
        example_values = "196.4, 232.4, 285, 315.6, 338.4, 308.7, 330.5, 332.3, 340.4, 350.9"

        self.values_text.delete(1.0, tk.END)
        self.values_text.insert(1.0, example_values)
        self.alpha_entry.delete(0, tk.END)
        self.alpha_entry.insert(0, "0.0625")

    def clear_data(self):
        """Очистка всех данных"""
        self.values_text.delete(1.0, tk.END)
        self.alpha_entry.delete(0, tk.END)

        # Очистка таблицы
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Очистка статистики
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, "Здесь будет отображаться статистика после расчета...")
        self.stats_text.config(state=tk.DISABLED)

        # Очистка графика
        self.ax.clear()
        self.ax.set_title("График появится после расчета", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Год")
        self.ax.set_ylabel("Значение")
        self.canvas.draw()

        self.df = None
        self.y = None
        self.trend_coeffs = None

        messagebox.showinfo("Очистка", "Все данные успешно очищены!")

    def calculate(self):
        """Выполнение расчета прогноза"""
        try:
            # Получение входных данных
            values_text = self.values_text.get(1.0, tk.END).strip()
            if not values_text:
                messagebox.showwarning("Внимание", "Введите исходные данные!")
                return

            values = [float(x.strip()) for x in values_text.split(",")]

            if len(values) != 10:
                messagebox.showerror("Ошибка", f"Нужно ровно 10 значений!\nВведено: {len(values)}")
                return

            # Получение параметра α
            alpha_text = self.alpha_entry.get().strip()
            if not alpha_text:
                messagebox.showwarning("Ошибка", "Введите параметр α!")
                return

            alpha = float(alpha_text)
            if alpha <= 0 or alpha >= 1:
                messagebox.showerror("Ошибка", "α должен быть в диапазоне: 0 < α < 1")
                return

            # Выполнение расчета
            self.df, self.trend_coeffs, self.y = calculate_forecast(values, alpha)

            # Обновление таблицы
            self.update_table()

            # Обновление статистики
            self.update_statistics(values, alpha)

            # Построение графика
            self.update_chart()

            # Переключение на вкладку с графиками
            self.notebook.select(self.chart_frame)

            messagebox.showinfo("Успешно", "✅ Расчет успешно завершен!")

        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Проверьте правильность данных:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Ошибка расчета", f"Ошибка при расчете:\n{str(e)}")

    def update_table(self):
        """Обновление таблицы с результатами"""
        # Очистка таблицы
        for item in self.tree.get_children():
            self.tree.delete(item)

        if self.df is None:
            return

        # Заполнение таблицы
        for _, row in self.df.iterrows():
            values = [
                int(row["Год"]),
                f"{row['S1']:.4f}",
                f"{row['S2']:.4f}",
                f"{row['S3']:.4f}",
                f"{row['A0']:.4f}",
                f"{row['A1']:.4f}",
                f"{row['A2']:.4f}",
                f"{row['Прогноз']:.2f}",
                f"{row['Ошибка']:.2f}",
                f"{row['Верхняя']:.2f}",
                f"{row['Нижняя']:.2f}"
            ]
            self.tree.insert("", tk.END, values=values)

    def update_statistics(self, values, alpha):
        """Обновление статистической информации"""
        if self.trend_coeffs is None:
            return

        a0, a1, a2 = self.trend_coeffs

        stats_text = f"""
{'=' * 60}
КОЭФФИЦИЕНТЫ КВАДРАТИЧНОГО ТРЕНДА
{'=' * 60}
A0 = {a0:.6f}
A1 = {a1:.6f}
A2 = {a2:.6f}

УРАВНЕНИЕ ТРЕНДА:
y = {a0:.4f} + {a1:.4f}·t + {a2:.4f}·t²

{'=' * 60}
ПАРАМЕТРЫ МОДЕЛИ
{'=' * 60}
Коэффициент сглаживания (α) = {alpha}
Количество исходных данных = {len(values)}
Период прогнозирования = 13 лет (2004-2016)

{'=' * 60}
СВОДНАЯ СТАТИСТИКА ПРОГНОЗА
{'=' * 60}
Минимальное значение: {self.df['Прогноз'].min():.2f}
Максимальное значение: {self.df['Прогноз'].max():.2f}
Среднее значение: {self.df['Прогноз'].mean():.2f}
Стандартное отклонение: {self.df['Прогноз'].std():.2f}

ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ:
Средняя ширина: {self.df['Ошибка'].mean():.2f}
Диапазон ширины: [{self.df['Ошибка'].min():.2f}, {self.df['Ошибка'].max():.2f}]

{'=' * 60}
ВРЕМЯ РАСЧЕТА: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}
"""

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text.strip())
        self.stats_text.config(state=tk.DISABLED)

    def update_chart(self):
        """Обновление графика"""
        if self.df is None or self.trend_coeffs is None:
            return

        # Очистка предыдущего графика
        self.ax.clear()

        a0, a1, a2 = self.trend_coeffs

        # Годы наблюдений (2004-2013)
        years_obs = np.arange(2004, 2014)

        # Годы прогноза (2004-2016)
        years_all = self.df["Год"].values

        # Значения тренда для всех годов
        t_all = np.arange(1, 14)  # t от 1 до 13
        trend_all = a0 + a1 * t_all + a2 * (t_all ** 2)

        # Прогнозные значения
        forecast_all = self.df["Прогноз"].values

        # Выбор типа графика
        chart_type = self.chart_type.get()

        if chart_type == "all":
            # Все данные
            self.ax.plot(years_obs, self.y, 'o-', linewidth=2.5, markersize=8,
                         label='Наблюдаемые данные', color=Colors.CHART_COLORS[0], alpha=0.9)

            self.ax.plot(years_all, trend_all, 's--', linewidth=2, markersize=5,
                         label='Квадратичный тренд', color=Colors.CHART_COLORS[1], alpha=0.8)

            self.ax.plot(years_all, forecast_all, 'D-', linewidth=2.5, markersize=6,
                         label='Прогноз (сглаживание)', color=Colors.CHART_COLORS[2], alpha=0.9)

            # Доверительные интервалы
            self.ax.fill_between(years_all,
                                 self.df["Нижняя"].values,
                                 self.df["Верхняя"].values,
                                 alpha=0.15, color=Colors.CHART_COLORS[2],
                                 label='Доверительный интервал')

            title = "Сравнение наблюдаемых данных, тренда и прогноза"

        elif chart_type == "forecast":
            # Только прогноз
            self.ax.plot(years_all, forecast_all, 'D-', linewidth=3, markersize=8,
                         label='Прогноз (сглаживание)', color=Colors.CHART_COLORS[2])

            # Доверительные интервалы
            self.ax.fill_between(years_all,
                                 self.df["Нижняя"].values,
                                 self.df["Верхняя"].values,
                                 alpha=0.2, color=Colors.CHART_COLORS[2],
                                 label='Доверительный интервал')

            title = "Прогнозные значения с доверительными интервалами"

        else:  # "trend"
            # Только тренд
            self.ax.plot(years_obs, self.y, 'o-', linewidth=2, markersize=7,
                         label='Наблюдаемые данные', color=Colors.CHART_COLORS[0], alpha=0.7)

            self.ax.plot(years_all, trend_all, 's-', linewidth=2.5, markersize=6,
                         label='Квадратичный тренд', color=Colors.CHART_COLORS[1])

            title = "Наблюдаемые данные и квадратичный тренд"

        # Настройка графика
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=Colors.PRIMARY)
        self.ax.set_xlabel('Год', fontsize=11, fontweight='bold', color=Colors.DARK)
        self.ax.set_ylabel('Значение показателя', fontsize=11, fontweight='bold', color=Colors.DARK)

        # Сетка
        self.ax.grid(True, alpha=0.2, linestyle='-')

        # Легенда
        self.ax.legend(loc='best', fontsize=10, framealpha=0.9, shadow=True)

        # Настройка делений на оси X
        self.ax.set_xticks(years_all[::2])
        self.ax.tick_params(axis='x', rotation=45)

        # Настройка цвета осей
        self.ax.spines['bottom'].set_color(Colors.GRAY)
        self.ax.spines['left'].set_color(Colors.GRAY)

        # Автоматическая настройка масштаба
        self.ax.autoscale_view()

        # Улучшенное расположение
        self.fig.tight_layout()
        self.canvas.draw()

    def export_excel(self):
        """Экспорт результатов в Excel"""
        if self.df is None:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта!")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile=f"прогноз_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            )

            if file_path:
                # Сохраняем основные результаты
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    self.df.to_excel(writer, sheet_name='Прогноз', index=False)

                    # Добавляем лист с исходными данными
                    if self.trend_coeffs is not None:
                        a0, a1, a2 = self.trend_coeffs
                        stats_df = pd.DataFrame({
                            'Параметр': ['Коэффициент сглаживания (α)', 'A0', 'A1', 'A2'],
                            'Значение': [self.alpha_entry.get(), a0, a1, a2]
                        })
                        stats_df.to_excel(writer, sheet_name='Параметры', index=False)

                messagebox.showinfo("Успешно", f"✅ Данные сохранены в файл:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{str(e)}")

    def save_chart(self):
        """Сохранение графика в файл"""
        if self.df is None:
            messagebox.showwarning("Предупреждение", "Нет графика для сохранения!")
            return

        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("PDF files", "*.pdf"),
                    ("SVG files", "*.svg")
                ],
                initialfile=f"график_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

            if file_path:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor=Colors.WHITE)
                messagebox.showinfo("Успешно", f"✅ График сохранен в файл:\n{file_path}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{str(e)}")

    def copy_to_clipboard(self):
        """Копирование данных в буфер обмена"""
        if self.df is None:
            messagebox.showwarning("Предупреждение", "Нет данных для копирования!")
            return

        try:
            # Формируем текстовое представление таблицы
            lines = ["Год\tS1\tS2\tS3\tA0\tA1\tA2\tПрогноз\tОшибка\tВерхняя\tНижняя"]

            for _, row in self.df.iterrows():
                line = f"{row['Год']}\t{row['S1']:.4f}\t{row['S2']:.4f}\t{row['S3']:.4f}\t" \
                       f"{row['A0']:.4f}\t{row['A1']:.4f}\t{row['A2']:.4f}\t" \
                       f"{row['Прогноз']:.2f}\t{row['Ошибка']:.2f}\t" \
                       f"{row['Верхняя']:.2f}\t{row['Нижняя']:.2f}"
                lines.append(line)

            # Копируем в буфер обмена
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(lines))

            messagebox.showinfo("Успешно", "✅ Данные скопированы в буфер обмена!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать данные:\n{str(e)}")


# -------------------------- ЗАПУСК ПРИЛОЖЕНИЯ --------------------------
def main():
    root = tk.Tk()
    app = ForecastApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()