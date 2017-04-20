# ML Tech HW1

## 1


$$x = 4$$

![](/home/tinray/tmp/MLT/hw1/1.png)


## 2 

The dual formulation of the hard margin SVM is

$$\min_{\alpha} \frac{1}{2} \sum_{n=1}^7 \sum_{m=1}^7 \alpha_n \alpha_m y_n y_m K(x_n, x_m) - \sum_{n=1}^N \alpha_n$$

subject to 

$$\sum_{n=1}^{N} y_n \alpha_n = 0; \alpha_n \geq 0$$

for $n = 1, 2, 3, 4, 5, 6, 7$.

The optimal $\alpha$ is

$$
\begin{bmatrix}
0 \\
0.00176 \\
0.00463 \\
0.00640 \\
0 \\
0 \\
0 \\
\end{bmatrix}
$$

$x_1, x_2, x_3$ are support vectors.

## 3

\begin{align*}
 & K(x, x') \\
=& (2 + x^T x)^2 \\
=& 4 + 4 x^T x + (x^T x)^2 \\
=& 4 + 4 x_1 x'_1 + 4 x_2 x'_2 + 2 x_1 x_2 x'_1 x'_2 + x_1^2 {x'_1}^2 + x_2^2 {x_2'}^2 \\
=& \Phi_2(x)^T \Phi_2(x) \\
\end{align*}

so 

$$\Phi_2(x) = (2, 2 x_1, 2 x_2, \sqrt{2} x_1x_2, x_1^2, x_2^2)$$

\begin{align*}
w &= \sum_{n=1}^{N} \alpha_n y_n \Phi_2(x_n) \\
  &= - 0.00176 \Phi_2(x_2) - 0.00463 \Phi_2(x_3) + 0.0640 \Phi_2(x_4) \\
  &= - 0.00176 \begin{bmatrix}
      2 \\
      6 \\
      -10 \\
      -15 \sqrt{2} \\
      9 \\
      25
  \end{bmatrix} - 0.00463 \begin{bmatrix}
      2 \\
      6 \\
      -2 \\
      -3 \sqrt{2} \\
      9 \\
      1 \\
  \end{bmatrix} + 0.00640 \begin{bmatrix}
      2 \\
      10 \\
      -4 \\
      -10 \sqrt{2} \\
      25 \\
      4
  \end{bmatrix} \\
 &= \begin{bmatrix}
         0.02698 \\
         0.10794 \\
         -0.05548 \\
         0.11021 \sqrt{2} \\
         0.22939 \\
         0.07873
    \end{bmatrix} \\
  b &= y_1 - w^T z_1 = -9.59585
\end{align*}

The hyperplane in $\mathcal{Z}$ space is $w^T z + b = 0$.


## 4

No, they should be different. Since the transformation function from $\mathcal{X}$ to $\mathcal{Z}$ is different.


## 5

Assume that the minimum can be attained when $w = w', b = b', \xi = \xi'$ where $\xi'_i < 0$ for some $i$.

Then for $\xi''$ where ${\xi''}_i = 0$,

$$\frac{1}{2} {w'}^T w' + C \sum_{n=1}^{N} {\xi''}_n^2 < \frac{1}{2} w'^T w' + C \sum_{n=1}^{N} {\xi'}_n^2$$

and 

$$y_n(w'^T x_n + b) \geq 1 - {\xi'}_n > 1 - {\xi''}_n$$

for $n = i$, since ${\xi'}_i < {\xi''}_i = 0$.

Therefore, we attain a better solution when $\xi = \xi''$, which is contradict to the assumption.

Thus, we cannot attain the optimal solution when $\xi_n < 0$. That is, the constrain $\xi_n \geq 0$ is not required.

## 6

\begin{align*}
  & \mathcal{L}((b, w, \xi), \alpha) \\
 =& \frac{1}{2} w^T w + C \sum_{n=1}^{N} \xi_n^2 + \sum_{n=1}^{N} \alpha_n (1 - \xi_n - y_n(w^Tx_n + b))
\end{align*}

## 7

\begin{align*}
  & \frac{\partial}{\partial b} \mathcal{L}((b, w, \xi), \alpha) \\
 =& \sum_{n=1}^{N} -\alpha_n y_n
\end{align*}

Constrain $\sum_{n=1}^{N} \alpha_n y_n = 0$,

$$\mathcal{L}((b, w, \xi), \alpha) = \frac{1}{2} w^T w + C \sum_{n=1}^{N} \xi_n^2 + \sum_{n=1}^{N} \alpha_n (1 - \xi_n - y_n w^Tx_n)$$

$$\frac{\partial}{\partial w} \mathcal{L}((b, w, \xi), \alpha) = 
w - \sum_{n=1}^{N} a_n y_n x_{n}
$$

Let $w = \sum_{n=1}^{N} a_n y_n x_{n}$,

\begin{align*}
  & \mathcal{L}((b, w, \xi), \alpha) \\
 =& \frac{1}{2} \lVert \sum_{n=1}^{N} a_n y_n x_{n} \rVert^2 + C \sum_{n=1}^{N} \xi_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \xi_n) - w^T \sum_{n=1}^{N} a_n y_n x_n \\
 =& -\frac{1}{2} \lVert \sum_{n=1}^{N} a_n y_n x_{n} \rVert^2 + C \sum_{n=1}^{N} \xi_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \xi_n)
\end{align*}

$$\frac{\partial}{\partial \xi_i} \mathcal{L}((b, w, \xi), \alpha) = 2C \xi_i - \alpha_i $$

Let $\xi_i = \frac{1}{2C} \alpha_i$,

\begin{align*}
  & \mathcal{L}((b, w, \xi), \alpha) \\
 =& -\frac{1}{2} \lVert \sum_{n=1}^{N} a_n y_n x_{n} \rVert^2 + \frac{1}{4C^2} \sum_{n=1}^{N} \alpha_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \frac{\alpha_n}{2C}) \\
 =& -\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_n a_m y_n y_m x_{n}^T x_m + \frac{1}{4C^2} \sum_{n=1}^{N} \alpha_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \frac{\alpha_n}{2C}) \\
\end{align*}


## 8

\begin{align*}
  & \mathcal{L}((b, w, \xi), \alpha) \\
 =& -\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_n a_m y_n y_m z_{n}^T z_m + \frac{1}{4C^2} \sum_{n=1}^{N} \alpha_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \frac{\alpha_n}{2C}) \\
 =& -\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_n a_m y_n y_m \phi(x_{n})^T \phi(x_m) + \frac{1}{4C^2} \sum_{n=1}^{N} \alpha_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \frac{\alpha_n}{2C}) \\
 =& -\frac{1}{2} \sum_{n=1}^{N} \sum_{m=1}^{N} a_n a_m y_n y_m K(x_n, x_m) + \frac{1}{4C^2} \sum_{n=1}^{N} \alpha_n^2 + \sum_{n=1}^{N}  \alpha_n (1 - \frac{\alpha_n}{2C}) \\
\end{align*}


## 9

The answer is **[b]**.

Let the matrix $K_1$ where ${K_1}_{ij} = K_1(x_i, x_j)$.

### [a]

The matrix $K$, where $K_{ij} = K(x_i, x_j)$, is not positive semi-definite when 

$$K_1 = \begin{bmatrix}0.9 & 0 \\0 & 0.9\end{bmatrix}$$

since for $a = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$

\begin{align*}
  & a^T K a \\
 =& a^T \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} a -
    a^T \begin{bmatrix} 0.9 & 0 \\ 0 & 0.9 \end{bmatrix} a \\
 =& -1.8 < 0
\end{align*}

### [b]

The matrix $K$, where $K_{ij} = K(x_i, x_j)$, is always positive semi-definite since $K$ is a all 1 matrix and it can be decomposed as

$$K = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix}^T$$.

### [c]

The matrix $K$, where $K_{ij} = K(x_i, x_j)$, is not positive semi-definite when the $K_1$ is same as **[a]**'s since for 

$$a = \begin{bmatrix} 1 \\ -10 \end{bmatrix}$$

\begin{align*}
  & a^T K a \\
 =& a^T \begin{bmatrix} 1 & 10 \\ 10 & 1 \end{bmatrix}a \\
 =& -99 < 0
\end{align*}

### [d]

The matrix $K$, where $K_{ij} = K(x_i, x_j)$, is not positive semi-definite when the $K_1$ is same as **[a]**'s since for 

$$a = \begin{bmatrix} 1 \\ -10 \end{bmatrix}$$

\begin{align*}
  & a^T K a \\
 =& a^T \begin{bmatrix} 1 & 100 \\ 100 & 1 \end{bmatrix}a \\
 =& -999 < 0
\end{align*}


## 10

Let $\alpha'$ be the optimal solution that minimizes 

$$\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m) - \sum_{n=1}^{N} \alpha_n$$

subject to

$$\sum_{n=1}^N y_n \alpha_n = 0; 0 \leq \alpha_n \leq C$$ 

for $n = 1, 2, \cdots, N$.

Then $\alpha'$ is also the optimal solution that minimizes

\begin{align}
 & \frac{1}{2} \frac{1}{p} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n y_m K(x_n, x_m) - \frac{1}{p} \sum_{n=1}^{N} \alpha_n \\
=& \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \frac{\alpha_n}{p} \frac{\alpha_m}{p} y_n  y_m p K(x_n, x_m) - \sum_{n=1}^{N} \frac{\alpha_n}{p} + \frac{q}{2} \sum_{n=0}^N y_n \frac{\alpha_n}{p} \sum_{m=0}^N y_m \frac{\alpha_m}{p} \\
 & \text{(since the constraint that $\sum_{n=1}^N y_n \alpha_n = 0$)} \\
=& \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \frac{\alpha_n}{p} \frac{\alpha_m}{p} y_n  y_m (p K(x_n, x_m) + q) - \sum_{n=1}^{N} \frac{\alpha_n}{p} \\
=& \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \frac{\alpha_n}{p} \frac{\alpha_m}{p} y_n  y_m \tilde{K}(x_n, x_m) - \sum_{n=1}^{N} \frac{\alpha_n}{p} \\
\end{align}

subject to

$$\sum_{n=1}^N y_n \frac{\alpha_n}{p} = 0; 0 \leq \frac{\alpha_n}{p} \leq \frac{C}{p} = \tilde{C}$$ 

for $n = 1, 2, \cdots, N$.

Thus, $\tilde{\alpha} = \frac{\alpha'}{p}$ is the optimal solution that minimizes

$$\frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N \alpha_n \alpha_m y_n  y_m \tilde{K}(x_n, x_m) - \sum_{n=1}^{N} \alpha_n$$

subject to

$$\sum_{n=1}^N y_n \alpha_n = 0; 0 \leq \alpha_n \leq \tilde{C}$$ 

The optimal $\beta', w'$ for original $C$ and $K$ is 

$$\beta' = C - \alpha'$$

and optimal $\tilde{\beta}$ for $\tilde{C}, \tilde{K}$ is

$$\tilde{\beta} = \tilde{C} - \tilde{\alpha} = \frac{1}{p}(C - \alpha')$$

Then the hyperplane for original $C, K$ is 

$$\{ x |\sum_{n=1}^N \alpha' y_n K(x_n, x) + C - \alpha' = 0 \}$$

while the hyperplane for $\tilde{C}, \tilde{K}$ is

\begin{align*}
 & \{ x |\sum_{n=1}^N \tilde{\alpha} y_n \tilde{K}(x_n, x) + \tilde{C} - \tilde{\alpha} = 0 \} \\
=& \{ x |\sum_{n=1}^N \frac{\alpha'}{p} y_n (pK(x_n, x) + q) + \frac{C}{p} - \frac{\alpha'}{p} = 0 \} \\
=& \{ x |\sum_{n=1}^N \alpha' y_n K(x_n, x) + \frac{C}{p} + q\sum_{n=1}^N \frac{\alpha'}{p} y_n - \frac{\alpha'}{p} = 0 \} \\
=& \{ x |\sum_{n=1}^N \alpha' y_n K(x_n, x) + \frac{C}{p} - \frac{\alpha'}{p} = 0 \} \\
\end{align*}

which is different from that of original one.

Therefore, for the dual of soft-margin support vector machine, using \tilde{K} along
with a new $\tilde{C} = \frac{C}{p}$ instead of $K$ with the original $C$ does not lead to an equivalent $g_{\mathrm{svm}}$ classifier.
