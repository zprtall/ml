import math

alpha = 0.01
w1, w2, b = 0.1, 0.1, 0.5
epochs = 1000

data = [
    ((-1, 0), 0), ((-1, 3), 0), ((-7, 2), 0), ((2, 4), 0),
    ((8, 5), 1), ((7, 8), 1), ((8, 7), 1), ((5, 5), 1),
]
n = len(data)

def sigmoid(s):
    return 1 / (1 + math.exp(-s))

def sigmoid_deriv(s):
    e = math.exp(-s)
    return e / ((e + 1) ** 2)

def epoch(w1, w2, b):
    loss = 0.0
    dw1 = dw2 = db = 0.0

    for (x1, x2), ti in data:
        s = x1 * w1 + x2 * w2 - b
        yi = sigmoid(s)
        loss += (yi - ti) ** 2

        g = 2 * (yi - ti) * sigmoid_deriv(s)
        dw1 += g * x1
        dw2 += g * x2
        db  -= g

    w1 -= alpha * dw1 / n
    w2 -= alpha * dw2 / n
    b  -= alpha * db  / n
    return w1, w2, b, loss / n

for e in range(epochs):
    w1, w2, b, loss = epoch(w1, w2, b)
    if e % 100 == 0:
        print(f"эпоха {e}: loss = {loss:.4f}")

print(f"\nитог: w1={w1:.3f}, w2={w2:.3f}, b={b:.3f}, loss={loss:.4f}")