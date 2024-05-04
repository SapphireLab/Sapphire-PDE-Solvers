# An Extreme Learning Machine-Based Method for Computational PDEs in Higher Dimensions

作者: Yiran Wang, Suchuan Dong
发表: Elsevier - Computer Methods in Applied Mechanics and Engineering 418 (2024) 116578
链接: [DOI](https://doi.org/10.1016/j.cma.2023.116578)
页数: 31
引用: 98

## Abstract·摘要

> We present two effective methods for solving high-dimensional partial differential equations(PDE) based on randomized neural networks.
> Motivated by the universal approximation property of this type of networks, both methods extend the extreme learning machine (ELM) approach from low to high dimensions.
> With the first method the unknown solution field in 𝑑 dimensions is represented by a randomized feed-forward neural network, in which the hidden-layer parameters are randomly assigned and fixed while the output-layer parameters are trained.
> The PDE and the boundary/initial conditions, as well as the continuity conditions (for the local variant of the method), are enforced on a set of random interior/boundary collocation points.
> The resultant linear or nonlinear algebraic system, through its least squares solution, provides the trained values for the network parameters.
> With the second method the high-dimensional PDE problem is reformulated through a constrained expression based on an Approximate variant of the Theory of Functional Connections (A-TFC), which avoids the exponential growth in the number of terms of TFC as the dimension increases.
> The free field function in the A-TFC constrained expression is represented by a randomized neural network and is trained by a procedure analogous to the first method.
> We present ample numerical simulations for a number of high-dimensional linear/nonlinear stationary/dynamic PDEs to demonstrate their performance.
> These methods can produce accurate solutions to high-dimensional PDEs, in particular with their errors reaching levels not far from the machine accuracy for relatively lower dimensions.
> Compared with the physics-informed neural network (PINN) method, the current method is both cost-effective and more accurate for high-dimensional PDEs.

## 1.Introduction·引言

> This work concerns the numerical approximation of partial differential equations (PDEs) in higher dimensions (typically beyond three).
> Mathematical models describing natural and physical processes or phenomena are usually expressed in PDEs.
> In a number of fields and domains, including physics, biology and finance, the models are naturally formulated in terms of high-dimensional PDEs.
> Well-known examples include the Schrodinger equation for many-body problems in quantum mechanics, the Black–Scholes equation for the price evolution of financial derivatives, and the Hamilton–Jacobi–Bellman (HJB) equation in dynamic programming and game theory [1,2].
> Development of computational techniques for PDEs is a primary thrust in scientific computing.
> In low dimensions, traditional numerical methods such as the finite difference, finite element (FEM), finite volume, and spectral type methods (and their variants), which are typically grid- or mesh-based, have achieved a tremendous success and are routinely used in computational science and engineering applications.
> For high-dimensional PDEs, on the other hand, these mesh-based approaches encounter severe challenges owing to the curse of dimensionality, because the computational effort/complexity involved therein grows exponentially with increasing problem dimension [3–6].

> In the past few years deep neural networks (DNN or NN) have emerged as a promising approach to alleviate or overcome the curse of dimensionality for solving high-dimensional PDEs [7–10].
> DNN-based methods usually compute the PDE solution in a mesh-free manner by transforming the PDE problem into an optimization problem.
> The PDE and the boundary/initial conditions are encoded into the loss function by penalizing their residual norms on a set of sampling points.
> The differential operators involved therein are typically computed by automatic differentiation.
> The loss function is minimized by an optimizer, usually based on some flavor of gradient descent type algorithms [11].
> Early works on NN-based methods for differential equations can be traced to the 1990s (see e.g. [12–14]).
> More recent prominent methods in this area include the physics-informed neural network (PINN) method [15], deep Galerkin method (DGM) [16], deep Ritz method [17], deep Nitsche method [18], deep mixed residual method [19], as well as other related approaches, variants and extensions (see e.g. [20–36], among others).
> Another approach for solving high-dimensional PDEs is to reformulate the problem using stochastic differential equations, thus casting the PDE problem into a learning problem.
> Representative techniques of this type include the deep backward stochastic differential equation (Deep BSDE) [1,37] and the forward–backward stochastic neural network method [38].
> Temporal difference learning has been employed in [39,40] for solving high-dimensional parabolic PDEs and partial integro-differential equations, which discretize the problem in time and represents the solution by a neural network at each time step.
> A data-driven method is developed in [33] to approximate the semi-global solutions to the HJB equations for high-dimensional nonlinear systems and to compute the optimal feedback controls.
> In [34] the generalization error bounds are derived for two-layer neural networks in the framework of deep Ritz method for solving two elliptic PDEs, and it is shown that the errors are independent of the problem dimension.
> We would also like to refer the reader to [41] for a recent review of NN-based techniques for high-dimensional PDEs.

> For the neural network-based techniques reviewed above for high-dimensional PDEs, all the weight/bias parameters in the neural network are trained and determined by an optimizer, which in most cases is Adam, L-BFGS or some related variant.
> Unlike these methods, in the current work we consider another type of neural networks for the computation of high-dimensional PDEs, referred to as randomized neural networks (or random-weight neural networks), in which a subset of the network parameters is assigned to random values and fixed (not trainable) while the rest of the network parameters are trained.

> Randomness has long been exploited in neural networks [42].
> Randomized neural networks can be traced to the un-organized machine by Turing [43] and the perceptron by Rosenblatt [44] in the 1950s.
> Since the early 1990s, methods based on randomized NNs have witnessed a strong resurgence and expansion [45,46], with prominent techniques widely applied and exerting a profound influence over a variety of areas [42,47].

> A simple strategy underlies randomized neural networks.
> Since it is extremely hard and expensive to optimize the full set of weight/bias parameters in the neural network, it seems sensible if a subset of the network parameters is randomly assigned and fixed, so that the resultant optimization problem of network training can become simpler, and in certain cases linear, hopefully without severely sacrificing the network’s achievable approximation capacity [48,49].
> When applied to different types of neural networks or under different configurations, randomization gives rise to several techniques, including the random vector functional link (RVFL) network [50–52], the extreme learning machine (ELM) [53,54], and the echo-state network [55,56], among others.

> We consider the extreme learning machine (ELM) approach for high-dimensional PDE problems.
> The original work on ELM was [53,57], developed for linear classification and regression problems with single hidden-layer feed-forward neural networks.
> This method has since found widespread applications in many fields [58,59].
> ELM is characterized by two ideas, randomly-assigned non-trainable (fixed) hidden-layer parameters, and trainable linear output-layer parameters determined by linear least squares method or by the pseudo-inverse of coefficient matrix [51,60].
> Randomized neural networks of the ELM type and its close cousin RVFL type, with a single hidden layer, are universal function approximators.
> Their universal approximation ability has been established by the theoretical studies of [52,54,61,62].
> In particular, the expected rate of convergence for approximating Lipschitz continuous functions has been provided by [52,62,63] (see also Section 2.2 below).

> The adoption of ELM for scientific computing, in particular for the numerical solution of differential equations, occurs only fairly recently.
> The existing works in this area have been confined to PDEs in low dimensions (primarily one or two spacial dimensions) or ordinary differential equations (ODEs) so far.
> Early works in this regard [64–66] have used polynomials (e.g.
> Chebyshev, Legendre, Bernstein) as activation functions for solving linear ODEs/PDEs.
> Subsequent contributions have explored other types of functions and made advances on a variety of fronts.
> While many studies are confined to linear ODE/PDE problems (see e.g. [67–74]), ELM-based methods for nonlinear PDEs/ODEs have been developed in e.g. [48,49,75–82] (among others).
> As has become clear from these studies, the ELM technique can produce highly accurate solutions to linear and nonlinear PDEs in low dimensions (and ODEs) with a competitive computational cost.
> For smooth solutions the ELM errors decrease exponentially as the number of degrees of freedom (number of training points, or number of trainable parameters) increases [49,75], reminiscent of the traditional high-order methods such as the spectral or spectral element techniques [83–88].
> Their errors can reach the level of machine accuracy as the degrees of freedom become large [48].
> In the presence of local complex features (e.g. sharp gradient) in the solution field, a combination of domain decomposition and ELM, referred to as local ELM (or locELM) in [75], will be critical to achieving a high accuracy [49].
> ELM-based methods have been compared extensively with the traditional numerical methods (e.g. classical FEM, high-order finite elements) and with the dominant DNN-based solvers (e.g.
> PINN/DGM) for low-dimensional PDE problems; see e.g. [48,75].
> ELM far outperforms the classical FEM, and also outperforms the high-order FEM markedly when the problem size is not very small [48].
> With a small problem size, the performance of ELM and high-order FEM is comparable, with the latter being slightly better [48].
> Here ‘‘outperform’’ refers to the ability of a method to achieve a better accuracy under the same computational cost or to incur a lower computational cost for the same accuracy.
> ELM also considerably outperforms DGM and PINN for low-dimensional problems [75].
> Very recently it has been shown by [80] that the ELM-based method exhibits a spectral accuracy for solving inverse PDE problems (in low dimensions) if the measurement data is noise-free, when the network is trained by nonlinear least squares or the variable projection algorithm [89].

> In the current paper we focus on the computation of high-dimensional PDEs with the ELM-based approach.
> There seems to be very little investigation in this aspect so far.
> A recent work related to this topic is [90], in which random feature neural networks are found to be able to approximate the functions with a convolutional structure efficiently (without curse of dimensionality).
> Since the solutions to linear Kolmogorov PDEs associated to exponential Levy models (e.g.
> Black–Scholes equation) can be expressed into this type of functions based on the Feynman–Kac formula, this work recasts the problem of learning the solution to linear Kolmogorov PDEs into a regression problem and employs random feature neural networks to approximate the solution data.
> We note that the method in [90] relies on an external Monte-Carlo solver to first generate the solution data to the Kolmogorov PDE in order to train the random feature neural network by regression.
> We also note the interesting theoretical aspect of [90].

> The technique developed in the current work computes high-dimensional PDEs in a ‘‘physics-informed’’ manner, which is self-contained and does not rely on any external PDE solver.
> The high-dimensional initial/boundary value problems considered here also involve more general PDEs and bounded domains.
> We are especially interested in the following question:
> - Is the ELM-type randomized neural network approach effective for computational PDEs in high dimensions?
>
> The objective of this paper is to present two ELM-based methods for solving high-dimensional PDEs, and to demonstrate with numerical simulations that these methods provide a positive answer to the above question, at least for the range of problem dimensions studied in this paper.

> The first method (termed simply ELM herein) extends the ELM technique and its local variant locELM developed in [75] (for low-dimensional problems) to linear and nonlinear PDEs in high dimensions.
> The solution field to the high-dimensional PDE problem is represented by a randomized feed-forward neural network, with its hidden-layer coefficients randomly assigned and fixed and its output-layer coefficients trained.
> Enforcing the PDE, the boundary and initial conditions on a random set of collocation points from the domain interior and domain boundaries gives rise to a linear or nonlinear algebraic system of equations about the trainable NN parameters.
> We seek a least squares solution to this algebraic system, attained by either linear or nonlinear least squares method, which provides the trained values for the network parameters.
> In the local variant of this method, the high-dimensional domain is decomposed along a maximum of  ( = 2 herein) directions, and the solution field on each sub-domain is represented by an ELM-type randomized neural network.
> We enforce the PDE, the boundary/initial conditions and appropriate continuity conditions across sub-domains on a set of random collocation points from each sub-domain, from the domain boundaries and from the shared sub-domain boundaries.
> The resultant linear or nonlinear algebraic system yields, by its least squares solution, the trained values for the network parameters of the local NNs.

> The second method (termed ELM/A-TFC herein) combines the ELM approach and an approximate variant of the theory of functional connections (TFC) for solving high-dimensional PDEs.
> TFC [91,92] provides a systematic approach for enforcing the boundary/initial conditions through a constrained expression (see e.g. [77,93]).
> However, the number of terms in TFC constrained expressions grows exponentially with respect to the problem dimension, rendering TFC infeasible for high-dimensional problems.
> By noting a hierarchical decomposition of the constrained expression, we introduce an approximate variant of TFC (referred to as A-TFC herein) that retains only the dominant terms therein.
> A-TFC avoids the exponential growth in the number of terms of TFC and is suitable for high-dimensional problems.
> On the other hand, since A-TFC is an approximation of TFC, its constrained expression does not satisfy the boundary conditions unconditionally for an arbitrary free function contained therein.
> However, the conditions for the free function of the A-TFC constrained expression in general involve functions of a simpler form, which is effectively a linearized form of those of the original boundary/initial conditions.
> A-TFC represents a trade-off.
> It carries a level of benefit of TFC for enforcing the boundary/initial conditions and is simultaneously suitable for high-dimensional PDEs.
> The ELM/A-TFC method uses the A-TFC constrained expression to reformulate the given high-dimensional PDE problem into a transformed problem about the free function contained in the expression.
> This free function is then represented by an ELM-type randomized neural network, and the reformulated PDE problem is enforced on a set of random collocation points.
> The least squares solution to the resultant algebraic system provides the trained values for the network parameters, thus leading to the solution for the free function.
> The solution to the original high-dimensional PDE problem is then computed based on the A-TFC constrained expression.

> Ample numerical simulations are presented to test these methods for a number of high-dimensional PDEs that are linear or nonlinear, stationary or time-dependent.
> The current method has also been compared with the PINN method for a range of problem dimensions.
> The numerical results show that the current methods exhibit a clear sense of convergence with respect to the number of training parameters and the number of boundary collocation points for high-dimensional PDEs.
> The rate of convergence is close to exponential for an initial range of parameter values (before saturation).
> These methods can capture the solutions to high-dimensional PDEs quite accurately, in particular with their errors reaching levels not far from the machine accuracy for comparatively lower dimensions.
> The error levels produced by these two methods are generally comparable, with ELM/A-TFC appearing slightly better in lower dimensions.
> On the other hand, ELM generally involves a smaller computational effort and cost than ELM/A-TFC.
> Compared with PINN, the current ELM method can achieve a significantly better accuracy under a markedly lower computational cost (network training time) for solving high-dimensional PDEs.

> The contributions of this paper lie in the ELM method and the ELM/A-TFC method presented herein for computing high-dimensional PDE problems.
> To the best of the authors’ knowledge, this seems to be the first physics-informed technique based on ELM-type randomized neural networks for solving high-dimensional PDEs.

> The methods presented in this paper are implemented in Python based on the Tensorflow and Keras libraries.
> The linear and nonlinear least squares methods are based on routines from the Scipy library.
> The numerical simulations are performed on a MAC computer (Apple M1 Chip, 8 cores, 8 GB memory, 250 GB hard disk, macOS Ventura) in the authors’ institution.

> The rest of this paper is organized as follows.
> In Section 2 we first briefly recall the theoretical result on ELM-type randomized NNs for function approximations in high dimensions, and then describe the ELM method and the ELM/A-TFC method for solving high-dimensional PDEs.
> In Section 3 we present extensive numerical simulations to test these two methods with several linear and nonlinear, stationary and dynamic PDEs for a range of problem dimensions.
> The current method is also compared with PINN.
> Section 4 concludes the presentation with a summary of the results and some further remarks.

## 2.Methodology·方法

### 2.1.Randomized Feed-Forward Neural Networks·随机前馈神经网络

### 2.2.Randomized NNs for High-Dimensional Function Approximation·随机神经网络用于高维函数逼近

### 2.3.Solving High-Dimensional PDEs with ELM·使用 ELM 解高维 PDE

### 2.4.Solving High-Dimensional PDEs by Combined ELM and Approximate Theory of Functional Connections (A-TFC)·使用 ELM/A-TFC 解高维 PDE

#### 2.4.1.TFC and Approximate TFC·TFC 和近似 TFC

#### 2.4.2.A-TFC embedded ELM·A-TFC 嵌入 ELM

## 3.Experiments·实验

### 3.1.Numerical Tests with the ELM Method·ELM 方法的数值测试
#### 3.1.1.Poisson Equation·泊松方程
#### 3.1.2.Nonlinear Poisson Equation·非线性泊松方程
#### 3.1.3.Advection Diffusion Equation·平流扩散方程
#### 3.1.4.Korteweg-de Vries Equation·科特韦格德-德维尔方程

### 3.2.Numerical Tests with the ELM/A-TFC Method·ELM/A-TFC 方法的数值测试
#### 3.2.1.Poisson Equation·泊松方程
#### 3.2.2.Nonlinear Poisson Equation·非线性泊松方程
#### 3.2.3.Heat Equation·热方程

### 3.3.Comparison with PINN·与 PINN 的比较

## 4.Conclusions·结论


> In this paper we have presented two methods for computing high-dimensional PDEs based on randomized neural networks.
> These methods are motivated by the theoretical result established in the literature that the ELM-type randomized NNs can effectively approximate high-dimensional functions, with a rate of convergence independent of the function dimension in the sense of expectations.

> The first method extends the ELM approach, and its local variant locELM, developed in a previous work for low-dimensional problems to linear/nonlinear PDEs in high dimensions.
> We represent the solution field to the high-dimensional PDE problem by a randomized NN, with its hidden-layer coefficients assigned to random values and fixed and its output-layer coefficients trained.
> Enforcing the PDE problem on a set of collocation points randomly distributed on the interior/boundary of the domain leads to an algebraic system of equations, which is linear for linear PDE problems and nonlinear for nonlinear PDE problems, about the ELM trainable parameters.
> By seeking a least squares solution to this algebraic system, attained by either a linear or a nonlinear least squares method, we can determine the values for the training parameters and complete the network training.
> ELM can be combined with domain decomposition and local randomized NNs for solving high-dimensional PDEs, leading to a local variant of this method.
> In this case, domain decomposition is performed along a maximum of two designated directions for a 𝑑-dimensional problem, and the PDE problem, together with appropriate continuity conditions, is enforced on the random collocation points on each sub-domain and the shared sub-domain boundaries.

> Compared with the ELM for low-dimensional problems, the difference of the method here for high-dimensional PDEs lies in at least two aspects.
> First, the collocation points employed for training the ELM network for high-dimensional PDEs are randomly generated on the interior and the boundaries of the domain (or the sub-domains), and the number of interior collocation points has little (essentially no) effect on the ELM accuracy in high dimensions.
> In contrast, for low-dimensional PDE problems the ELM neural network is trained largely on grid-based collocation points (e.g. uniform grid points, or quadrature points), and the number of interior collocation points critically influences the ELM accuracy.
> Second, with the local variant of ELM (plus domain decomposition) for solving high-dimensional PDEs, the domain is only decomposed along a maximum of  directions, where  is a prescribed small integer ( = 2 in this paper), so as for the method to be feasible in high dimensions.
> This is an issue not present for low-dimensional PDEs.

> The second method (ELM/A-TFC) combines the ELM approach and an approximate variant of TFC (A-TFC) for solving high-dimensional PDEs.
> While TFC provides a systematic approach to enforce the boundary/initial conditions, the number of terms involved in TFC constrained expression grows exponentially as the problem dimension increases, rendering it infeasible for high-dimensional problems.
> By noting that the TFC constrained expression can be decomposed into a hierarchical form, we introduce the A-TFC by retaining only the dominant terms in the constrained expression.
> A-TFC avoids the exponential growth in the number of terms of TFC and is feasible for high-dimensional PDEs.
> On the other hand, the A-TFC constrained expression does not unconditionally satisfy the boundary/initial conditions for an arbitrary free function in the expression.
> However, the conditions that the free function in the A-TFC constrained expression needs to fulfill, in order to satisfy the boundary/initial conditions, involve functions of simpler forms, which in some sense can be considered as an effective linearization of those involved in the original boundary/initial conditions.
> A-TFC carries a level of benefit of TFC for enforcing the boundary/initial conditions and is simultaneously suitable for high-dimensional problems.
> With the ELM/A-TFC method, we reformulate the high-dimensional PDE problem using the A-TFC constrained expression, and attain a transformed problem about the free function involved in the A-TFC expression.
> We represent this free function by ELM, and determine the ELM trainable parameters by the linear or nonlinear least squares method in a fashion analogous to the first method.
> After the free function is determined by the ELM network, the solution field to the original high-dimensional PDE problem is then computed by the A-TFC constrained expression.

> The two methods have been tested numerically using a number of linear/nonlinear stationary/dynamic PDEs for a range of problem dimensions.
> The method has also been compared with the PINN method.
> We have the following observations from these numerical results:
> - Both the ELM method and the ELM/A-TFC method produce accurate solutions to high-dimensional PDEs, in particular with their errors reaching levels not far from the machine accuracy for relatively lower dimensions.
> - Both methods exhibit a clear sense of convergence with respect to the number of trainable parameters and the number of boundary collocation points.
> Their errors decrease rapidly (exponentially or nearly exponentially) for an initial range of parameter values (before saturation).
> - The number of interior collocation points appears to have a minimal (essentially no) effect on the accuracy of ELM and ELM/A-TFC for high-dimensional PDEs.
> - For a given PDE, the problem becomes more challenging to compute with increasing dimension, in the sense that the errors of both methods in higher dimensions generally appear somewhat worse than in lower dimensions, at least with the range of parameter values tested in this work.
> - The error levels obtained by the ELM method and the ELM/A-TFC method are generally comparable, with ELM/A-TFC appearing slightly better in lower dimensions.
> On the other hand, the ELM/A-TFC method generally involves a larger computational effort and cost than ELM, due to the A-TFC constrained expression.
> - The current method exhibits a clear advantage compared with PINN for solving high-dimensional PDEs, and achieves a significantly better accuracy under markedly smaller training time than the latter.

> The simulation results signify that the ELM-based methods developed herein are effective for computational PDEs in high dimensions.