import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 1. Grafik ##

age = [10, 20, 30, 40, 50, 60, 70, 75]
weight = [20, 60, 80, 85, 86, 87, 70, 90]

age = np.array(age)
weight = np.array(weight)

plt.plot(age, weight, "g")  # g renk
plt.xlabel("age")
plt.ylabel("weight")
plt.title("1. Grafik")
plt.show()

## 2. Grafik ##

numpy_array_1 = np.linspace(0, 10, 20)
numpy_array_2 = numpy_array_1 ** 3
plt.plot(numpy_array_1, numpy_array_2, "g--")  # g-- g*-
plt.title("2. Grafik")
plt.show()

## 3. Grafik ##

plt.subplot(1, 2, 1)
plt.plot(numpy_array_1, numpy_array_2, "g--")
plt.title("3. Grafik")  # tek birinde oluyo ya genel değil başlık
plt.subplot(1, 2, 2)
plt.plot(numpy_array_2, numpy_array_1, "r*-")
plt.title("3. Grafik")
plt.show()

## 4. Grafik - FIGURE ##

figure = plt.figure()  # figsize=(8, 6) w-h
axes = figure.add_axes([0.1, 0.1, 0.8, 0.8])  # 0.8 ler boyut
axes.plot(numpy_array_1, numpy_array_2, color="#FFAA97", alpha=0.5, linewidth=4.0, linestyle="--", marker="o",
          markersize=10, markerfacecolor="green")  # alpha saydamlık  -. : -- *-
axes.plot(numpy_array_2, numpy_array_1, color="#3188CE", alpha=1, linewidth=1.0)  # alpha saydamlık
axes.set_xlabel("X axes")
axes.set_ylabel("Y axes")
axes.set_title("4. Grafik")
plt.show()

## 5. Grafik - FIGURE ##

figure_2 = plt.figure(figsize=(8, 6))

axes_1 = figure_2.add_axes([0.1, 0.1, 0.7, 0.7])
axes_2 = figure_2.add_axes([0.2, 0.4, 0.3, 0.3])

axes_1.plot(numpy_array_1, numpy_array_2, "g")
axes_1.set_xlabel("X axes")
axes_1.set_ylabel("Y axes")
axes_1.set_title("5. Grafik - Ana Başlık")

axes_2.plot(numpy_array_2, numpy_array_1, "b")
axes_2.set_xlabel("X axes")
axes_2.set_ylabel("Y axes")
axes_2.set_title("5. Grafik - Alt Başlık")
plt.show()

## 6. Grafik - SUBPLOT ##

(figure, axes) = plt.subplots(1, 2)

for axe in axes:
    axe.plot(numpy_array_1, numpy_array_2, "g")
    axe.set_xlabel("X axes")
plt.tight_layout()
plt.show()

## 7. Grafik ##

figure_3 = plt.figure(dpi=200)  # kalite
axes_3 = figure_3.add_axes([0.1, 0.1, 0.9, 0.9])
axes_3.plot(numpy_array_1, numpy_array_2, label="numpy_array_2 ** 2")
axes_3.legend(loc=2)
plt.show()

figure_3.savefig("figure_3.png", dpi=200)  # kaydetmek

## 8. Grafik - Scatter ##

# scatter
plt.scatter(numpy_array_1, numpy_array_2)
plt.show()


# hist
numpy_array_3 = np.random.randint(0, 100, 50)
plt.hist(numpy_array_3)
plt.show()

# boxplot
plt.boxplot(numpy_array_3)
plt.show()
