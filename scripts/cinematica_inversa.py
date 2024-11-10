import math

px, py, pz = 0.8675, 0, 0.1
px, py, pz = -0.85675, 0, 0.00519



px, py, pz = -0.2575, 0.9, 0.05+0.05
px, py, pz = 0.6175, -0.3, 0.05+0.05
px, py, pz = 0.3175, -0.675, 0.05+0.05

px, py, pz = 0.31749963760375977, -0.6750001311302185, 0.10000009015202523

l1 = 0.31 - 0.05
l2 = 0.467003
l3 = 0.400499

r = math.sqrt(math.pow(px, 2) + math.pow(py, 2))

try:
    beta = math.acos((math.pow(l2, 2)+math.pow(r, 2)-math.pow(l3, 2))/(2*l2*r))
except ValueError:
    beta = 0

q1 = math.atan2(py, px) - beta

q2 = q1 + math.asin((r/l3)*math.sin(beta))

q3 = l1 - pz

q1_grados = math.degrees(q1)
q2_grados = math.degrees(q2)

# Mostrar los resultados
print(f"q1 (grad) = {q1_grados}")
print(f"q2 (grad) = {q2_grados}")
print(f"q3 (m) = {q3}")

