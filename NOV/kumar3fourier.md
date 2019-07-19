
## The Fourier Transform

In order to specify practical properties of discrete time systems, such as low pass filtering or high pass filtering, it is necessary to transform the complex z-plane to the real-frequency axis,$\omega$.  Specifically, the region of the complex z-plane that is used in the transformation is the unit circle, specified by the rgion $z=e^{j\omega}$. The resulting transform is the Discrete-Time Fourier Transform (DTFT).

However, due to the need for more applicable easily computable transform, the Discrete Fourier Transform (DFT) is introduced, which is very homogeneous in both forward (time to frequency), and inverse (frequency to time) formulations.  The crowning moment in the evolution of DSP came when the Fast Fourier (FFT) was discovered by Cooley and Tukey in 1965, which is essentially a fast algorithm to compute the DFT, makes it possible to achieve real-time audio and video processing.

### Discrete-Time Fourier Transform (DTFT)
The frequency response of a system is very important to define practical property of the system, such as low pass or high-pass filtering. It can be obtained by considering the system function H(z) on the unit circle.  Similarly for a discrete-time sequence x(n), we can define the z-tranform X(z) on the unit circle as follows:
$$X(e^{j\omega})=X(z)|_{z=e^{j\omega}}=\sum_{n=-\infty}^\infty x(n)e^{-j\omega n} - - - -\dots(1)$$

The function $X(e^{j\omega})$ or $X(\omega)$ is also called the Discrete Time Fourier Transform (DTFT) of the discrete-time signal x(n). The inverse DTFT is defined by the following integral:
$$x(n)=\frac{1}{2\pi}\int_{-\pi}^pi X(\omega)e^{j\omega n}d\omega - - - -\dots(2)$$

For all values of n.  The significance of the integration operation in equation (2) will be clear after discussing the periodicity property of the DTFT in the next section.

#### Properties of the Discrete Time Fourier Transform (DTFT)
A concise list of DTFT traform properties are given in table 1 below
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
    <td class="tg-yw4l">Periodicity</td>
    <td class="tg-yw4l">x(n)</td>
    <td class="tg-yw4l">$X(\omega)=X(\omega+2m\pi)$, for integer m</td>
  </tr>
  <tr>
    <td class="tg-yw4l">2</td>
    <td class="tg-yw4l">Convolution</td>
    <td class="tg-yw4l">$x(n)\ast h(n)$</td>
    <td class="tg-yw4l">$X(\omega)H(\omega)$</td>
  </tr>
  <tr>
    <td class="tg-yw4l">3</td>
    <td class="tg-yw4l">Time shift</td>
    <td class="tg-yw4l">$x(n-n_0)$</td>
    <td class="tg-yw4l">$X(\omega)e^{-j\omega n_0}$</td>
  </tr>
  <tr>
    <td class="tg-yw4l">4</td>
    <td class="tg-yw4l">Frequency Scaling</td>
    <td class="tg-yw4l">$e^{j\omega_0n}x(n)$</td>
    <td class="tg-yw4l">$X(\omega-\omega_0)$</td>
  </tr>
  <tr>
    <td class="tg-yw4l">5.</td>
    <td class="tg-yw4l">Time reversal</td>
    <td class="tg-yw4l"><br>$x(-n)$<br></td>
    <td class="tg-yw4l">$X(-\omega)$</td>
  </tr>
</table>

#### Example
Determine the Discrete-Time Fourier Transform (DTFT) of the following time functions

1. $x(n)=\sum_{k=0}^4\delta(n-k)$
2. $x(n)=(0.5)^{|n|}$



#### Solution
1. From equation (1)
    $$\begin{aligned}X(e^{j\omega})&=X(z)|_{z=e^{j\omega}}=\sum_{n=-\infty}^\infty x(n)e^{-j\omega n} \\
    &=\sum_{n=-\infty}^\infty [\delta(n)+\delta(n-1)+\delta(n-3)+\delta(n-4)]e^{-j\omega n} \\
    &=1+e^{-j\omega}+e^{-j2\omega}+e^{-j3\omega}+e^{-j4\omega} \\
    &=\frac{e^{-j5\omega}-1}{e^{-j\omega}-1}\text{, using finite Geometric series formula}
    \end{aligned} $$
2. Similrarly:
    $$\begin{aligned}X(e^{j\omega})&=X(z)|_{z=e^{j\omega}}=\sum_{n=-\infty}^\infty x(n)e^{-j\omega n} \\
    &=\sum_{n=-\infty}^-1 0.5^{|n|}e^{-j\omega n} \\
    &=\sum_{n=0}^\infty 0.5^{n}e^{-j\omega n} +\sum_{n=-\infty}^{-1} 0.5^{-n}e^{-j\omega n} \\
    &=\sum_{n=0}^\infty \left(0.5e^{-j\omega}\right)^{n} +\sum_{n=1}^\infty \left(0.5e^{j\omega}\right)^{n} \\
     &=\frac{1}{1-0.5e^{-j\omega}}+\frac{0.5e^{j\omega}}{1-0.5e^{j\omega}}
    \end{aligned} $$


#### Example
Determine the inverse Fourier Transform,x(n) of the function $X(e^{j\omega}), whose magnitude and phase are described as follows:

Magnitude
$$\left|X(e^{j\omega}

1. $x(n)=\sum_{k=0}^4\delta(n-k)$
2. $x(n)=(0.5)^{|n|}$



#### Solution
1. From equation (1)
    $$\begin{aligned}X(e^{j\omega})&=X(z)|_{z=e^{j\omega}}=\sum_{n=-\infty}^\infty x(n)e^{-j\omega n} \\
    &=\sum_{n=-\infty}^\infty [\delta(n)+\delta(n-1)+\delta(n-3)+\delta(n-4)]e^{-j\omega n} \\
    &=1+e^{-j\omega}+e^{-j2\omega}+e^{-j3\omega}+e^{-j4\omega} \\
    &=\frac{e^{-j5\omega}-1}{e^{-j\omega}-1}\text{, using finite Geometric series formula}
    \end{aligned} $$
2. Similrarly:
    $$\begin{aligned}X(e^{j\omega})&=X(z)|_{z=e^{j\omega}}=\sum_{n=-\infty}^\infty x(n)e^{-j\omega n} \\
    &=\sum_{n=-\infty}^-1 0.5^{|n|}e^{-j\omega n} \\
    &=\sum_{n=0}^\infty 0.5^{n}e^{-j\omega n} +\sum_{n=-\infty}^{-1} 0.5^{-n}e^{-j\omega n} \\
    &=\sum_{n=0}^\infty \left(0.5e^{-j\omega}\right)^{n} +\sum_{n=1}^\infty \left(0.5e^{j\omega}\right)^{n} \\
     &=\frac{1}{1-0.5e^{-j\omega}}+\frac{0.5e^{j\omega}}{1-0.5e^{j\omega}}
    \end{aligned} $$


#### Analog Frequency and Digital Frequency

The fundamental relation between the analog frequency, $\Omega$, and the digital frequency, $\omega$, is given by the following relation:
$$\omega=\Omega T - - - -\dots(3a)$$
or alternatively
$$\omega=\Omega/f_s - - - -\dots(3b)$$

where T is the sampling period, in sec., and $f_s=1/T$ is the sampling frequency in Hz.

In this important transformation however, one should note the folloing points.

1. The unit $\Omega$ is radian/sec., whereas, the unit $\omega$ is just radians.
2. The analog frequency, $\Omega$, represents the actual physical frequency of the basic analog signal, for example, an audio signal (0 to 4 kHz) or a video signal (0 to 4 MHz).  The digital frequency, $\omega$, is the transformed frequency from equation 3 and can be considered as a mathematical frequenc, corresponding to a digital signal.



```R

```
