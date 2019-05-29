
## LTI Systems

Linearity implies superposition, and time-invariance implies that properties of the system do not change with time.  Two of the most important concepts associated with discrete-time LTI systems are linear convolution and difference equations.

### Linear Convolution

Linear convolution is a natural process of LTI systems.  It defines the input-output relation of the system and is defined as:
$$y(n)=x(n)\ast h(n)$$
where $\ast$ symbol denotes the convolution process, x(n) is the system input, y(n) is the system output, and h(n) is the impulse response of the system.  The impule response is the output of the system, when the input $x(n)=\delta(n)$, the unit impulse.  The actual convolution process is defined:

$$y(n)=\sum_{k=-\infty}^\infty x(k)h(n-k)- - - -\dots(1)$$

for all values of n.  Convolution can be performed using the sliding tape method, as shown below, or more practically using computer software, as will be described below.

#### Example
Determine the linear convolution of the two discrete-time sequences, x(n) and h(n), given by
$$\begin{aligned}x(n)&=\begin{bmatrix}1&1&1&1\end{bmatrix} \\
                h(n)&=\begin{bmatrix}1&2&3\end{bmatrix}
    \end{aligned}$$

#### Solution

The sliding tape method can be done by hand and calculation, if the number of points in both sequences is quite small.  The procedure is as follos
- Write the sequence x(m), h(m), and h(-m) as shown below.  The sequence h(-m) is obtained by mirroring the sequence h(m) about the m=0 axis.  Then the dot product of the vectors x(m) and h(-m) gives the convolution output y(0).
$$\begin{aligned}
    x(m)&=\begin{bmatrix}0&0&0&1&1&1&1\end{bmatrix} \\
    h(m)&=\begin{bmatrix}0&0&0&1&2&3&0\end{bmatrix} \\
    h(-m)&=\begin{bmatrix}0&3&2&1&0&0&0\end{bmatrix};y(0)=1 
\end{aligned}$$

- The zeros are introduced in x(m) and h(m) sequences to include the negative time axis, which is required to generate h(-m) sequence.
- Similarly, the next term in the table below, h(1-m), is obtained by shifting (-m) by one step to the right.  The dot product of the vectors x(m) and h(1-m) gives the convolution output y(1).
$$h(1-m)=\begin{bmatrix}0&0&3&2&1&0&0\end{bmatrix};y(1)=3$$

- The process is continued until the output y(n) remains at zero
$$\begin{aligned}
    x(2-m)&=\begin{bmatrix}0&0&0&3&2&1&0\end{bmatrix};y(2)=6 \\
    h(3-m)&=\begin{bmatrix}0&0&0&0&3&2&1\end{bmatrix};y(3)=6 \\
    h(4-m)&=\begin{bmatrix}0&0&0&0&0&3&2\end{bmatrix};y(4)=5 \\
    h(5-m)&=\begin{bmatrix}0&0&0&0&0&0&3\end{bmatrix};y(5)=3 
\end{aligned}$$

- Any more shift in the sequence h(-m) will result in a zero output Hence the output vector is:
$$y(n)=\begin{bmatrix}1&3&6&6&5&3\end{bmatrix}$$
- Note that the length of the output vector y(n)=[length of x(n)]+[length of the h(n)]-1=4+3-1=6.  This is the general law of linear convolution.

### Linear Constant Coefficient Difference Equation
The difference equation is another fundamental relation between relation between the input and output of LTI systems.  It describes the input-output process of any discrete-time system, such as a high-pass or low pass filtering.  The difference equation takes the folloing form:
$$a_0y(n)+a_1y(n-1)+\dots+a_Ny(n-N)=b_0x(n)+b_1x(n-1)+\dots+b_Mx(n-M)$$
or compactly as follows:
$$\sum_{k=0}^Na_ky(n-k)=\sum_{k=0}^Mb_kx(n-k) - - - - (2)$$
Important properties of the difference equations are:
- The order of the difference equation, N where $N\ge M$.
- The differene equation can be solved recursively to obtain the output y(n), for a given $n=n_0$, if the initial conditions, $y(n),n<n_0$ are known.
- The coefficients $a_i, i=0,1,2,\dots,N,$ and $b_i,i=0,1,2,\dots,M$ are constant, real numbers and define the properties of the system.  For example, one set of coefficients could generate a low pass filter, while a different set of coefficients could generate a high pass filter.

#### Example
Consider a discrete-time system with input x(n) and output y(n), which is described by the folloing difference equation:
$$y(n)=2y(n-1)+x(n)$$
If the input $x(n)=\delta(n)$, determine the output y(n) for the time range $0\le n \le 5$, given the initial condition that y(n)=0, for n<0.

#### Solution
The difference equation can be solved recursively, starting with n=0;
$$\begin{aligned}y(0)&=2y(-1)+x(0)=2(0)+1=1 \\
    y(1)&=2y(0)+x(1)=2(1)+0=2  \\
    y(2)&=2y(1)+x(2)=2(2)+0=4  \\
    y(3)&=2y(2)+x(3)=2(4)+0=8  \\
    y(4)&=2y(3)+x(4)=2(8)+0=16  \\
    y(5)&=2y(4)+x(5)=2(16)+0=32  \end{aligned}$$

### Introduction to Z Transform and System Function H(z)
In the previous section of this chapter, we introduced two fundamental properties of LTI systems, viz., linear convolution and difference equation.  The link between these two fundamental aspects of LTI systems is provided by the z-transform.  The z-transform is a very important time-to-frequency domain transformation of the basic discrete-time signal x(n).  The z-transform is defined by the equation:
$$X(z)=\sum_{n=-\infty}^\infty x(n)z^{-n} - - - -\dots(3)$$
where the variable z is complex, and the function X(z) is defined on the complex plane. A concise list of z-transform properties is given in table 1 below
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;}
.tg .tg-yw4l{vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l"></th>
    <th class="tg-yw4l">Property</th>
    <th class="tg-yw4l">f(n)</th>
    <th class="tg-yw4l">F(z)</th>
  </tr>
  <tr>
    <td class="tg-yw4l">1</td>
    <td class="tg-yw4l">Linearity</td>
    <td class="tg-yw4l">$Ax_1(n)+Bx_2(n)$, for constants A, B</td>
    <td class="tg-yw4l">$AX_1(z)+BX_2(z)$</td>
  </tr>
  <tr>
    <td class="tg-yw4l">2</td>
    <td class="tg-yw4l">Convolution</td>
    <td class="tg-yw4l">$x(n)\ast h(n)$</td>
    <td class="tg-yw4l">X(z)H(z)</td>
  </tr>
  <tr>
    <td class="tg-yw4l">3</td>
    <td class="tg-yw4l">Time shift</td>
    <td class="tg-yw4l">$x(n-n_0)$</td>
    <td class="tg-yw4l">X(z)z^{n_0}</td>
  </tr>
  <tr>
    <td class="tg-yw4l">4</td>
    <td class="tg-yw4l">Frequency Scaling</td>
    <td class="tg-yw4l">$z_0^nx(n)$</td>
    <td class="tg-yw4l">$X(z/z_0)$</td>
  </tr>
  <tr>
    <td class="tg-yw4l">5.</td>
    <td class="tg-yw4l">Differentiation</td>
    <td class="tg-yw4l">nx(n)</td>
    <td class="tg-yw4l">$-z\frac{dX(z)}{dz}$</td>
  </tr>
</table>
**Table 1:** Z-Transform Theorems

Utilizing property(2) from table 1 in equation (1), we get the transfomed equation
$$Y(z)\sum_{k=0}^Na_kz^{-k}=X(z)\sum_{k=0}^Mb_kz^{-k} - - - -\dots(4)$$
Comparing equations (3) and (4) we get the following system function
$$H(z)=\frac{Y(z)}{X(z)}=\frac{\sum_{k=0}^Mb_kz^{-k}}{\sum_{k=0}^Na_kz^{-k}}- - - -\dots(5)$$

#### Properties of the System Function H(z)
The system function or transfer function H(z) can be expressed in the following forms:
##### Polynomial Form
This form can be obtained by expanding equation (5) to yield
$$H(z)=\frac{b_0+b_1z^{-1}+b_2z^{-2}+\dots+b_Mz^{-M}}{a_0+a_1z^{-1}+a_2z^{-2}+\dots+a_Nz^{-N}}- - - -\dots(6)$$

##### Pole-Zero Form and Stability of the System
This form can be obtained by factorizing the numerator and denominator polynomials in equation (6) to yeild
$$H(z)=\frac{b_0(1-c_1z^{-1})(1-c_2z^{-1})\dots(1-c_Mz^{-1})}{a_0(1-d_1z^{-1})(1-d_2z^{-1})\dots(1-d_Nz^{-1})}- - - -\dots(7)$$

The M roots of the numerator polynomial $c_1, c_2,\dots,c_M$ are the zeros, and the N roots of the denominator polynomial  $d_1, d_2,\dots,d_M$ are the poles of the system function.

The poles of the system function define the stability of the system in the complex z-plane.  If all the poles of the system function lie inside the unit circle, hence satisfying $|d_k|<1$ for k=1,2,...,N, then the system is unconditionally stable.  For an example system function $H(z)=\frac{(z-2)}{z(z-0.5)(z-0.8)}$, the poles or the roots of the denominator polynomial, are located at z=0, 0.5, and 0.8.  Since all the poles lie inside the unit circle, or have a magnitude of less than 1, the system H(z) is unconditionally stable.

A special case arises when one or more poles lies on the unit circle, with |z|=1, and all other poles inside the unit circle.  We define such a system as marginally stable, whose stability is limited by the input.

##### Example
A casual linear time-invariant system has the system function:
$$H(z)=\frac{(1-1.5z^{-1}-z^{-2})(1+0.9z^{-1})}{(1-z^{-1})(1+0.7jz^{-1})(1-0.7jz^{-1})}$$

1. Write the difference equation that relates the input and output of the system.
2. Is the system stable?

##### Solution
1> Expanding the numerator and denominator factors, the system function is
$$H(z)=\frac{1-0.6z^{-1}-2.35z^{-2}-0.9z^{-3}}{1-z^{-1}+0.49z^{-2}-0.49z^{-3}}=\frac{Y(z)}{X(z)} $$
Cross multiplying and taking the inverse z-transform gives:
$$\begin{aligned} y(n)-y(n-1)-0.49y(n-2)-0.49y(n-3) \\ =x(n)-0.6x(n-1)-2.35x(n-2)-0.9x(n-3) \end{aligned}$$

2> The poles of the system function are located at z=1, z=+0.7j, and z=-0.7j.  Since one of the poles is located on the unit circle, and all other poles lie inside the unit circle, the system is marginally stable.

##### System Frequency Response H($e^{j\omega}$)
The frequency response of the system is very important to define the practical property of the system, such as low pass or high pass filtering.  It can be obtain by considering the system function H(z) on the unit circle shown in Figure 1 which corresponds to the $z=e^{j\omega}$ circle.  Hence, from equation (5) the frequency response is

$$H(e^{j\omega})=H(z)|_{z=e^{j\omega}}- - - - \dots(8)$$

![Figure 1](https://selene.hud.ac.uk/u1273400/images/seg_media/lti0.PNG)
**Figure 1:** Complex z-plane with the unit circle.

##### Example
An Nth order comb filter has the system function $H(z)=1-z^{-N}$
1. Determine the pole-zero locations of H(z).
2. Evaluate the impulse response h(n).
3. Determine the frequency response (magnitude and phase) of the system

##### Solution
The system function is
$$\begin{aligned} H(z)&=1-z^{-N} \\ &=\frac{z^{-N}-1}{z^{-N}} \end{aligned}$$

1> From the system function, the poles and zeros can be obtained.  N zeros at
$$ \begin{matrix} z^N=1=e^{j2\pi k}, & k=0,1,\dots,N-1 \\ \implies z=e^{j2\pi k/N}, & k=0,1,\dots,N-1 \end{matrix}$$
and N poles at z=0.

2> The impulse respone can be obtained by taking the inverse z-transform of the system function H(z):
$$h(n)=\delta(n)-\delta(n-N)$$

3> The frequency response can also be obtained from the system function H(z) by using equation (8)

$$H(e^{j\omega})=H(z)|_{z=e^{j\omega}}=1-e^{-j\omega N}$$

which can be simplified to yield the final result

Since the frequency response of the system $H(e^{j\omega})$ is period in $2\pi$ radians, it is sufficient to define both magnitude and phase responses of the system in the interval $-\pi\le\omega\le\pi$ radians.

### Important LTI System types
The fundamental properties of LTI systems directly affect the behaviour of practical electrical components such as filters, amplifiers, oscillators, and antennas.  Some commonly used systems are described briefly below.
#### Inverse System
As the name implies, the inverse system $H_i(z)$ of a given system H(z) is defined as:
$$H_i(z)=\frac{1}{H(z)} - - - -\dots(9)$$

Inverse systems are utilized in audio and video processing to recover signals coming through noisy channels.  However, the inverse system may not be stable, even if the orignal system is stable.  This is because the zeros of the system H(z) are the poles of the system $H_i(z)$.  In order to overcome this problem, we would require a minimum phase system.

#### All-pass systems
As the name implies, an all-pass system has a frequency response magnitude that is independent of $\omega$. A stable system function of the form:
$$H_{ap}(z)=\frac{z^{-1}-a^*}{1-az^{-1}} - - - -\dots(10)$$
has the frequency response
$$\begin{aligned}H_{ap}(e^{-j\omega})&=\frac{e^{-j\omega}-a^*}{1-ae^{-j\omega}} \\
&=e^{-j\omega}\frac{1-a^*e^{j\omega}}{1-ae^{-j\omega}}\end{aligned} - - - -\dots(11)$$
which implies that the magnitude response $|H_{ap}(e^{-j\omega}|=1$

#### Minimum phase systems
A minimum phase system has both its poles and zeros inside the unit circle.  Hence in audio and video processing units, the inverse system can be designed as a reciprocal of a minimum phase system as follows:
$$H_i(z)=\frac{1}{H_{min}(z)} - - - -\dots(12)$$

This will ensure that the inverse system is stable.  The following points are very important in design of stable inversefilters in practical communication systems:
- In communication systems, signals often pass through noisy random channels.  In order to compensate for the effects of a noisy system $H_{noisy}(z)$, we design an inverse filter
$$H_i(z)=\frac{1}{H_{noisy}(z)} - - - - \dots(13)$$
    This inverse filter can basically cancel out the effect of the noisy system.
- However, since the inverse filter could be unstable, we make use of an important theorem, which states that:  For any specific rational system function H(z), the minimum-phase system $H_{min}(z)$ exists and can be derived using the general theorem that $H(z)=H_{min}(z)H_{ap}(z)$.
- Hence we can expand the noisy system function as:$H_{noisy}(z)=H_{min}(z)H_{ap}(z)$ and design a stable inverse system as follows.
$$H_i(z)=\frac{1}{H_{min}(z)} - - - -\dots(14)$$

Since the magnitude response of an all-passs sytem is a constant (which can be adjusted to unity), the magnitude respose of the inverse filter in equation (12) is the same as that of the filter in equation (9).  The process of creation of the minimum phase system from the original system is clearly explained in the example below.


```R

```
