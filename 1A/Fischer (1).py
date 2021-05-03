import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# getting the data and classify them based on classes
data = pd.read_csv("dataset_FLD.csv", header=None)
points_pos = np.where(data.iloc[:, -1].values == 1)
points_neg = np.where(data.iloc[:, -1].values == 0)

# calculating mean m1 and m2
l1 = len(data.iloc[points_pos])
l2 = len(data.iloc[points_neg])
m1 = sum(data.iloc[points_pos].values[:, :-1]) / l1
m2 = sum(data.iloc[points_neg].values[:, :-1]) / l2
m1 = m1.reshape(len(m1), 1)  # transpose getting the dimension 3*1
m2 = m2.reshape(len(m2), 1)  # transpose

# plotting the points in 3-D positive points in red and negative points in blue
X = data.iloc[points_pos].values[:, 0]
Y = data.iloc[points_pos].values[:, 1]
Z = data.iloc[points_pos].values[:, 2]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X, Y, Z, color='red')
X1 = data.iloc[points_neg].values[:, 0]
Y1 = data.iloc[points_neg].values[:, 1]
Z1 = data.iloc[points_neg].values[:, 2]
ax.scatter(X1, Y1, Z1, color='blue')
plt.title("data points")
plt.show()

#  calculating Sw and vector w
dim = int(len(m1))
S_pos = np.zeros((dim, dim), dtype=float)  # initializing the matrix of dimension d*d with zeroes
S_neg = np.zeros((dim, dim), dtype=float)
for i in range(len(data)):
    if data.iloc[i, -1:].values == 1:
        S_pos += np.dot(np.transpose(data.iloc[i, :-1].values - m1), np.asarray(data.iloc[i, :-1].values - m1))
    else:
        S_neg += np.dot(np.transpose(data.iloc[i, :-1].values - m2), np.asarray(data.iloc[i, :-1].values - m2))
S_pos /= l1
S_neg /= l2
Sw = S_neg + S_pos
print("Sw : \n", Sw, "\n" )
w = np.dot(np.linalg.inv(Sw), m1 - m2)  # inverse of Sw d*d dimension and m1-m2 is d*1 hence w is d*1
print("w: \n",w,"\n")
w_unit = w / (np.square(w).sum())**0.5
print("unit vector along w : \n", w_unit, "\n")

#  transforming the points into one dimension
res_vec = np.zeros(len(data), dtype=float)
for j in range(len(data)):
    res_vec[j] = np.dot(np.transpose(w), data.iloc[j, :-1].values)  # wTX

#  finding the normal distributions mean and variance
points_p = res_vec[points_pos]
points_n = res_vec[points_neg]
mean_p = np.mean(points_p)
mean_n = np.mean(points_n)
std_d_p = np.std(points_p)
std_d_n = np.std(points_n)

#  finding the intersection point
#  it is a solution of a quadratic equation
a0 = 1 / 2 * ((1 / std_d_p ** 2) - (1 / std_d_n ** 2))
a1 = (mean_n / std_d_n ** 2) - (mean_p / std_d_p ** 2)
a2 = 1 / 2 * (mean_p ** 2 / std_d_p ** 2 - mean_n ** 2 / std_d_n ** 2) - np.log(std_d_n / std_d_p)
solution = np.roots([a0, a1, a2])


# threshold point should lie in between the means
if max(mean_n, mean_p) >= solution[0] >= min(mean_n, mean_p):
    threshold_point = solution[0]
else:
    threshold_point = solution[1]
print("threshold value: ", threshold_point,"\n")

#  plotting w
x = np.arange(-1.25, 1.25, 0.05)
y = (-w[0] - w[1] * x) / w[2]
plt.plot(x, y)
y1 = (-w[0] - w[1] * threshold_point) / w[2]

# plotting the discriminant line
y2 = np.arange(0, 5, 0.1)
x2 = (w[1] * y + w[2] * threshold_point-w[1]*y1) / w[2]
plt.plot(x2, y2, color='green')

#  unit vector of discriminant  line
discriminant_line = np.zeros([len(w), 1])
discriminant_line[0] = w[2]*threshold_point-w[1]*y1
discriminant_line[1] = -w[2]
discriminant_line[2] = w[1]
dis_line_unit = discriminant_line / (np.square(discriminant_line).sum())**0.5
print("discriminant line unit vector \n",dis_line_unit,"\n")

#  plotting the normal distribution
points_p.sort()
points_n.sort()
plt.plot(points_p, stats.norm.pdf(points_p, mean_p, std_d_p), color='red')
plt.plot(points_n, stats.norm.pdf(points_n, mean_n, std_d_n), color='blue')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Normal Distribution of points")
plt.show()

#  plotting the threshold point
x = threshold_point
y = 1
plt.plot(x, y, marker='v', color="black")

# plotting the points in 1D
plt.scatter(points_p, l1 * [1], color='red')
plt.scatter(points_n, l2 * [1], color='blue')
plt.title("points in 1d")
plt.show()

#  plotting the separator plane along with points
X = data.iloc[points_pos].values[:, 0]
Y = data.iloc[points_pos].values[:, 1]
Z = data.iloc[points_pos].values[:, 2]
fig = plt.figure(num=1, clear=True)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(X, Y, Z, color='red')
X1 = data.iloc[points_neg].values[:, 0]
Y1 = data.iloc[points_neg].values[:, 1]
Z1 = data.iloc[points_neg].values[:, 2]
ax.scatter(X1, Y1, Z1, color='blue')
(x, y) = np.meshgrid(np.arange(-10, 10, 1), np.arange(-10, 10, 1))
z = (threshold_point - w[0] * x - w[1] * y) / w[2]  # wTx=t is the separator plane
ax.plot_surface(x, y, z, color='yellow')
ax.set(xlabel='x', ylabel='y', zlabel='z')
fig.tight_layout()
plt.show()

# unit vector in 3d
sep_plane = np.zeros([len(w)+1, 1])
sep_plane[0] = -threshold_point
sep_plane[1:4] = w
sep_plane_unit = sep_plane/np.linalg.norm(sep_plane)
print("Sep_unit: \n",sep_plane_unit,"\n")


# calculating accuracy
if mean_p < threshold_point:  # positive points are to the left and wtX<threshold for positive points
    c1 = 1
    c2 = 0
else:
    c1 = 0
    c2 = 1
prediction = np.zeros((len(data), 1))
given_class = data.iloc[:, -1].values
for i in range(len(data)):
    if res_vec[i] <= threshold_point:
        prediction[i] = c1
    else:
        prediction[i] = c2
count = 0
for j in range(len(data)):
    if prediction[j] == given_class[j]:
        count = count + 1;

accuracy = count / len(data)
print("The accuracy for the dataset: ", accuracy * 100, " % ")

