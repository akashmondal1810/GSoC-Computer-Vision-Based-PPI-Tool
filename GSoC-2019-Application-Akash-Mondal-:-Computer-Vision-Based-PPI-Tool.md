# About me  
## Username and Contact Information  
**Name** :            Akash Mondal

**University** :      [Indian Institute of Technology (IIT), Kharagpur](http://iitkgp.ac.in)

**Email** :           akashmondalcivil@iitkgp.ac.in  

**Gitter Handle** :   akashmondal1810 

**Github Username** : [akashmondal1810](https://github.com/akashmondal1810/)  

**Time Zone** : IST (UTC +5:30)

## Personal Background  
Hello, I am Akash Mondal, a third year undergraduate student at IIT Kharagpur, India. I am pursuing a degree in Civil Engineering. I work on Ubuntu 16.04 LTS. I am proficient in C, C++, Python. I am moderately proficient in Julia.  
## Relevant Courses  
* Machine Learning  
* Deep Learning
* Probability and Statistics

## Contribution to OrdinaryDiffEq.jl
I started contributing to OrdinaryDiffEq.jl in early February. I implemented Native Adams Bashforth and Adams Moulton methods and I learned a lot about code base of OrdinaryDiffEq.jl.
* (**Merged**) `Native Adams-Bashforth (3 steps)-Moulton (2 steps) method` [#256](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/pull/265)
*  (**Merged**) `added adams_bashforth (4 steps) and adams_moulton (3 steps) methods`[#269](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/pull/269)
*  (**Merged**) `added Adams-Bashforth (5 Steps) and Adams-Moulton (4 Steps) Methods`[#272](https://github.com/JuliaDiffEq/OrdinaryDiffEq.jl/pull/272)

## Experience with Computer Vision  
I have been using Julia for last two months. I like Julia because of its performance and multiple dispatch feature. Julia produces similar speed to C, while having a code similar to Python or Matlab. 

# The Project
## The Problem and Motivations  
* An efficient integrator must be able to change the step size and order, in order to control the errors of the method and to ensure stability properties.
* If the ODE (or system of ODEs) has a stiff part and an imaginary part, Using a purely implicit method may be acceptable in this scenario, or depending on the problem, may dissipate the imaginary part to an unwanted extent.Further, one could envision a situation where the imaginary part is nonlinear and the stiff part linear in this scenario, using an implicit method is the worst of both worlds, we end up solving a nonlinear problem, and it isn’t one we want to solve (possibly due to dissipation). On the other hand, explicit methods are often unacceptable for stiff problems due to the limited size of their stability regions.

The aim of this project is to add variable step size, variable order variable step size multistep methods, Implicit-Explicit (IMEX) multistep and IMEX Runge-Kutta Methods. 

# The Plan  
I propose to implement following methods :
1. Variable Step Size Multistep Methods
2. Variable Order Variable Step Size Multistep Methods  
3. IMEX Multistep Methods  
4. IMEX Runge-Kutta Methods  

**Stretch Goals**  
1. Implementation of variable time step variable order BDF method  
2. Implementation of variable time step variable order SBDF method

## 1. Variable Step Size Multistep Methods  
An efficient integrator must be able to change the step size. However, changing the step size with multistep methods is difficult since the formulas require the numerical approximations at equidistant points. There are in principle two possibilities for overcoming this difficulty [1]:
1. Use polynomial interpolation to reproduce the starting values at the new (equidistant) grid.  
2. Construct methods which are adjusted to variable grid points.  
##### Variable Step Size Adams Methods  
The explicit Adams method is :  
![5 4](https://user-images.githubusercontent.com/23627932/37771542-e8604be2-2dfd-11e8-8329-4e005c14ddff.png)  
And the extension of this method to variable step size is :  
![5 55](https://user-images.githubusercontent.com/23627932/37771575-116e7c66-2dfe-11e8-9a02-197ba75feeba.png)  
And the variable step size implicit Adams methods :  
![5 7](https://user-images.githubusercontent.com/23627932/37771603-2a99c8e4-2dfe-11e8-8ce0-6a26829d7ba4.png)  
here ![pn 1](https://user-images.githubusercontent.com/23627932/37771724-773d81d6-2dfe-11e8-9ec8-2fd15174f348.png) can be obtained by the explicit Adams method.  
Now in order to implement these methods, first we have to get values of `g_{j,n}`, `ϕ_{j,n}` and `ϕstar_{j,n}`, using Recurrence Relations for `g_{j,n}`, `ϕ_{j,n}` and `ϕstar_{j,n}` required coefficients can be calculated.  
### Recurrence Relations  
Relation between `ϕ` and `ϕstar`  
![recurrel](https://user-images.githubusercontent.com/23627932/37772149-c8d6dfc8-2dff-11e8-8ccd-74aab37f6f63.png)  
And for `β_{j,n}`  
![beta](https://user-images.githubusercontent.com/23627932/37811814-327f07c0-2e82-11e8-8220-651527af891b.png)   
And for `g_{j,n}`   
![gjc](https://user-images.githubusercontent.com/23627932/37811817-34fdbab4-2e82-11e8-9e21-d92e23cce3f0.png)  
where `c_{j,q}` can be calculated by this relation :  
![c](https://user-images.githubusercontent.com/23627932/37811819-3633581c-2e82-11e8-8fef-588537f61f93.png)  
All above formulas are described in [1].  
Following functions calculate coefficients using these relations [6] :  
```julia
# function for calculating g
function g_coefs!(g, c, dt, t, t_np1, n, k)
    for i = 1:k
      for q = 1:((k)-(i-1))
        if i == 1
          c[i,q] = 1/q
        elseif i == 2
          c[i,q] = 1/q/(q+1)
        else
          c[i,q] = c[i-1,q] - c[i-1,q+1] * dt/(t_np1-t[n-(i-1)+1])
        end
      end
      g[i]=c[i,1]
    end
    return nothing
end

# function for calculating ϕ and ϕstar
function ϕ_and_ϕstar_coefs!(ϕstar_n, ϕ_n, ϕstar_nm1, β, dy_n, t, t_np1, n, k)
    for i in k
      if i == 1
        β[i] = 1
        ϕ_n[i] = dy_n
        ϕstar_n[i] = dy_n
      else
        β[i] = β[i-1] * (t_np1 - t[n-(i-1)+1]) / (t[n] - t[n-(i-1)])
        ϕ_n[i] =  ϕ_n[i-1] - ϕstar_nm1[i-1]
        ϕstar_n[i] = β[i] * ϕ_n[i]
      end
    end
    return nothing
end

# function for calculating ϕ_{n+1,j}
function ϕ_coefs!(ϕ_np1, ϕstar_n, dy_np1, k)
    for i  in k
      if i == 1
          ϕ_np1[i] = dy_np1
      else
        ϕ_np1[i] = ϕ_np1[i-1] - ϕstar_n[i-1]
      end
    end
    return nothing
end
```

## 2. Variable Order Variable Step Size Multistep Methods   
#### Determining Optimal Step Size  
Staring multistep methods with order one and very small step sizes and therefore self-starting. Suppose after successful numerical integration until `x_n` if further step with step size `h_n` and order `k + 1` is taken, which yields the approximation `y_n+1` to `y(x_n+1)`.  For this step to be successful it must satisfy following equation   
![eq7 4](https://user-images.githubusercontent.com/23627932/37773683-3dec977c-2e04-11e8-8804-0e8547db78ac.png)  
where `LE_k+1` can be computed as  
![7 3](https://user-images.githubusercontent.com/23627932/37774029-215b8cac-2e05-11e8-962d-0f79aeb97b45.png)  
After the successful step, next optimal step size would be    
![7 5](https://user-images.githubusercontent.com/23627932/37774093-485fc264-2e05-11e8-8ebc-334a6c975ff5.png)  
One possible implementation with some modification :  
```julia
# Here k is the order of the method and h is step size
function chooseNextStep(le_k, k, h)
  if le_k < 2^(-k-2)
    return 2 * h
  elseif le_k < 1/2
    return h
  else
    return h * max(1/2, min(9/10, (1/2*le_k)^(k+1)))
  end
end
```
#### Determining Optimal Order  
After determining optimal step size, for determining the optimal order, there are essentially two strategies for selecting the new order. One can choose the order `k + 1` either such that the local error estimate is minimal, or such that the new optimal step size is maximal.    
After performing a step with order `k + 1`, order is reduced by one if the following condition satisfy  
![7 7](https://user-images.githubusercontent.com/23627932/37774162-86bf534e-2e05-11e8-99e7-24cdd0363a48.png)  
And increase in the order if the following condition satisfy  
![7 000](https://user-images.githubusercontent.com/23627932/37774246-b7bd60e4-2e05-11e8-9319-95fc390c615e.png)  
Here we can calculate `LE_{k+2, n+1}` by following equation :  
![lek 2](https://user-images.githubusercontent.com/23627932/37774388-0e9758e8-2e06-11e8-8901-1babf7007ef4.png)  
## 3. IMEX Multistep Methods  
If the ODE (or system of ODEs) has a stiff part and non-stiff part   
![imex1](https://user-images.githubusercontent.com/23627932/37777477-a6793288-2e0d-11e8-9b05-9a20678eea41.png)  
then purely implicit method may be acceptable in this scenario as explicit methods are unacceptable for stiff problems due to the limited size of their stability regions but one could envision a situation where the imaginary part is nonlinear and the stiff part linear in this scenario, using an implicit method is the worst of both worlds we end up solving a nonlinear problem, and it isn’t one we want to solve.  
This type of problem can be solved using a class of methods called Implicit-Explicit methods or IMEX methods in which we pick an implicit method for stiff part and an explicit method for non-stiff part.  
To solve non-stiff and stiff part separately, We can define problem as `SplitODEProblem`, in which one part `f1` will be stiff and `f2` will be non-stiff.  

```julia
# f1 -> g (stiff part)
# f2 -> f (non-stiff part)
prob = SplitODEProblem(f1,f2,u0,tspan)
```  
Now, In `perform_step!()` we can get `f1` by `f1 = integrator.f.f1` and similarly `f2` by `f2 = integrator.f.f2`.  
```julia
if typeof(integrator.f) <: SplitFunction
    f1 = integrator.f.f1
    f2 = integrator.f.f2
else
    error("type of prob should be SplitODEProblem")
end
```
And do implicit solving for `f1` and explicit solving for `f2`.  
Now, for solving implicit equation system, we can use Simplified Newton Iterations method (Quasi-Newton method) [2].  
![newton](https://user-images.githubusercontent.com/23627932/37864166-e2c15fd4-2f90-11e8-9c06-8d3726b7a382.png)  
And it is already implemented in `KenCarp and other implicit methods`.  
Here I am giving example -> how IMEX form of `KenCarp3` is working (important part and what it does).  
Implemented `KenCarp` methods are both IMEX and implicit SDIRK methods.  
when a user gives ODE as `SplitODEProblem` it switches to IMEX otherwise SDIRK by checking the following condition  
```julia
if typeof(integrator.f) <: SplitFunction
    f = integrator.f.f1
    f2 = integrator.f.f2
else
    f = integrator.f
end
```  

Here, assigning `integrator.f.f1` to `f` makes implementation easy because same code can be used for both methods for implicit part with some conditions.  
Now steps below are for calculation of `W (I - hA*J)` (in Simplified Newton method).  
```julia
if typeof(uprev) <: AbstractArray
    J = ForwardDiff.jacobian(uf,uprev)
    W = I - γdt*J
else
    J = ForwardDiff.derivative(uf,uprev)
    W = 1 - γdt*J
end
``` 
Now for solving implicit equation for `z₂` : 
1. Initial guess for z₂ `z₂ = z₁`
2. One step of simplified newton method and then checking condition for `do_newton`.
```julia
η = max(cache.ηold,eps(eltype(integrator.opts.reltol)))^(0.8)
do_newton = integrator.success_iter == 0 || η*ndz > κtol
```
3. Now run it in loop until any condition fails (do_newton, max_iter, convergence).
```julia
while (conditions)
    u = tmp + γ*z₂
    b = dt*f(u, p, tstep) - z₂
    dz = W\b
    z₂ = z₂ + dz
    # here convergence and do_newton conditions
end
``` 

Now calculating the solution of explicit part and combining with the solution of implicit part of IMEX method.
```julia
if typeof(integrator.f) <: SplitFunction
    z₃ = z₂
    u = tmp + γ*z₂
    k2 = dt*f2(u,p,t + 2γ*dt)
    tmp = uprev + a31*z₁ + a32*z₂ + ea31*k1 + ea32*k2
else
    z₃ = α31*z₁ + α32*z₂
    tmp = uprev + a31*z₁ + a32*z₂
end
```

IMEX Multistep methods to be implemented are described in [3] and [5].  

#### 1. Crank-Nicolson Admas Bashforth (CNAB)  
This is a second order method also known as **ABCN**  
The formula for this method is as follow :  
![imex2](https://user-images.githubusercontent.com/23627932/37778919-20e38fde-2e11-11e8-9fb3-de261c1c1538.png)   
This method can be easily implemented as the solution to the implicit part already implemented as `Trapezoid method` only thing to do is remove the adaptivity.  
#### 2. Crank-Nicolson Leapfrog (CNLF)  
This is a second-order method.  
The formula for this method is as follow :  
![imex3](https://user-images.githubusercontent.com/23627932/37779736-20b3a024-2e13-11e8-8f39-bb6e6ecf7cb2.png)  
In this method, Implicit part is same as above only explicit part is different. So doing required change in explicit part we can get CNLF from ABCN.  
#### 3. Adaptive Step Size Second Order BDF method  
The Advancing formula of the adaptive step size BDF2 is as follow [7]:   
![bdf2](https://user-images.githubusercontent.com/23627932/37910043-287bfe54-312a-11e8-83f8-574112cf302c.png)  
being `w_{n+1} = h_{n+2}/h_{n+1}`, `h_{n+2} = t_{n+2}-t_{n+1}` and `h_{n+1} = t_{n+1}-t_{n}`.  
And the expression of the local truncation error of the adaptive step
size BDF2 is :  
![lte2](https://user-images.githubusercontent.com/23627932/37910045-28d38c46-312a-11e8-89a4-a991aaf31b11.png)  
being `h_{n} = t_{n}-t_{n-1}`.  
Determining new step size :  
The new step size is determined on basis of the local error
estimation.  
New step size can be calculated by the method mentioned in variable order variable step size multistep method.  

#### 4. Second Order SBDF method
The formula for this method is as follow :  
![2__sbdf](https://user-images.githubusercontent.com/23627932/37873375-668a18a8-3039-11e8-8c79-b6ae3b6b58e4.png)  
Here k is step size.  

#### 5. Third Order SBDF method
In this method, implicit part is third order BDF.
The formula for this method is as follow :   
![3](https://user-images.githubusercontent.com/23627932/37879351-40a63a8e-3095-11e8-985f-d4792fafb488.png)  

#### 6. Fourth Order SBDF method  
This is 4th order 4-step method.
The formula for this method is as follow :  
![4](https://user-images.githubusercontent.com/23627932/37879350-40742346-3095-11e8-9e75-eb0c6835cab8.png)  

## 4. IMEX Runge-Kutta Methods  
Here also, same philosophy: pick an explicit scheme for `f` and an implicit one for `g`.  
I will implement IMEX Runge-Kutta schemes described in [4].  In this paper, schemes are divided into two classes.  
1. Those whose last internal stage is identified with the solution at the next time instance (i.e., at the end of the time step).
2. Those where an additional quadrature is used at the end of the step.  

**Note** : The first class is particularly good for highly-stiff problems and the scheme is identified as (3,4,3) (3 internal stages for the implicit formula, 4 stages for the explicit and a combined accuracy of order 3).  

In all scheme, For `g` (stiff) an implicit s-stage DIRK scheme will be used and for `f` (non-stiff) `σ = (s+1)-stage` explicit scheme will be used.  

General formula for these IMEX Runge-Kutta Methods is :   
![generalfo](https://user-images.githubusercontent.com/23627932/37767115-fac3485c-2dee-11e8-8f2a-b397d8b9652b.png)  
**Instances of this IMEX RK family of schemes to be implemented are** :  
##### 1. Forward-backward Euler (1, 1, 1)  
The pair of backward and forward Euler schemes  
![2 1](https://user-images.githubusercontent.com/23627932/37767772-4252f666-2df1-11e8-88a3-13c893809565.png)  
This yields the linear one step IMEX  
![eq1](https://user-images.githubusercontent.com/23627932/37767822-6fbf1e0e-2df1-11e8-814c-6732639dadd3.png)  
##### 2. Forward-backward Euler (1, 2, 1)  
![2 2](https://user-images.githubusercontent.com/23627932/37767906-b8b6526c-2df1-11e8-899d-7dd4dbe113e5.png)   
This yields  
![eee](https://user-images.githubusercontent.com/23627932/37767974-f066708e-2df1-11e8-89fb-127e9331ea76.png)  

##### 3. Implicit-explicit midpoint (1, 2, 2)  
The Euler pairs are first-order accurate and the (1,1,1) scheme has well-known disadvantages when `g = 0`. The following pair of implicit-explicit schemes:  
![2 3](https://user-images.githubusercontent.com/23627932/37768026-2a1f4fee-2df2-11e8-9487-92cfc7cfe47d.png)  
correspond to applying explicit midpoint for `f` and implicit midpoint for `g`. It is second-order accurate
because the two schemes from which it is composed are each second-order accurate.

##### 4. A third-order combination (2, 3, 3)  
Two stage, third order DIRK scheme  
![2 4 1](https://user-images.githubusercontent.com/23627932/37767106-f8b65284-2dee-11e8-86b1-f96e7d5b1d08.png)  
with ![sqrt](https://user-images.githubusercontent.com/23627932/37768183-b5ab51a2-2df2-11e8-97ca-64b4611443d6.png)  

The corresponding third-order explicit Runge-Kutta scheme (ERK) is  
![2 4 2](https://user-images.githubusercontent.com/23627932/37767107-f90e03d0-2dee-11e8-9d15-f1e8cee0e1e7.png)  
with ![sqrt](https://user-images.githubusercontent.com/23627932/37768183-b5ab51a2-2df2-11e8-97ca-64b4611443d6.png)  
The resulting IMEX combination is third-order accurate.  

##### 5. L-stable, two-stage, second-order DIRK (2, 3, 2)  
A two-stage, second-order DIRK scheme is  
![2 5 1](https://user-images.githubusercontent.com/23627932/37767083-eaf51f36-2dee-11e8-96a4-e4320af48e7d.png)  
with ![gamma5](https://user-images.githubusercontent.com/23627932/37768382-6b084898-2df3-11e8-9214-52f4a8d22fb1.png)    
The corresponding three-stage second-order ERK is  
![2 5 2](https://user-images.githubusercontent.com/23627932/37768460-b7208bf0-2df3-11e8-82cd-46f057ed54c4.png)  
with ![2 5eq](https://user-images.githubusercontent.com/23627932/37768499-e7665f56-2df3-11e8-868c-3df5ea252635.png)    
The resulting IMEX combination is second-order accurate.  

##### 6. L-stable, two-stage, second-order DIRK (2, 2, 2)  
Second order scheme is  
![2 66](https://user-images.githubusercontent.com/23627932/37768769-c8405572-2df4-11e8-8bf3-e0de7684175c.png)  
with ![gamma5](https://user-images.githubusercontent.com/23627932/37768382-6b084898-2df3-11e8-9214-52f4a8d22fb1.png) and ![gama and del](https://user-images.githubusercontent.com/23627932/37768702-93f50d76-2df4-11e8-82ed-602f0f31d789.png)  

##### 7.  L-stable, three-stage, third-order DIRK (3, 4, 3) 
 A three-stage, third-order DIRK scheme is  
![2 7 1](https://user-images.githubusercontent.com/23627932/37767100-f77fe3f8-2dee-11e8-9959-6f1d5579ca9c.png)  
The corresponding four-stage, third-order ERK scheme is  
![2 7 2](https://user-images.githubusercontent.com/23627932/37767097-f68de116-2dee-11e8-82f5-9dac118b8710.png)  
The resulting IMEX combination is third-order accurate.   
##### 8. A four-stage, third-order combination (4, 4, 3)   
In this method schemes are  
![2 8](https://user-images.githubusercontent.com/23627932/37766357-b9331eaa-2dec-11e8-954f-179e8c6d678f.png)  
The resulting IMEX combination is third-order accurate.  


# Answers to listed questions  
`1. What do you want to have completed by the end of the program?`  
By the end of the program, I want to have implementation of Variable Step Size Multistep Methods,
Variable Order Variable Step Size Multistep Methods, IMEX Multistep Methods and IMEX Runge-Kutta Methods (mentioned in the plan).  
`2. Who’s interested in the work, and how will it benefit them?`  
There are many fields where these methods can be useful for example fluid mechanics, fluid dynamics, bio.

`3. What are the potential hurdles you might encounter, and how can you resolve them?`  
In order to implement variable time step variable order BDF and SBDF methods, I will have to understand CVODE/LSODE because methods used in CVODE are variable-order, variable-step multistep methods.For nonstiff problems, CVODE includes the Adams-Moulton formulas, with variable order. For stiff problems, CVODE includes the Backward Differentiation Formulas (BDFs) variable order. I will also have to understand methods described in [8].

`4. How will you prioritize different aspects of the project like features, API usability, documentation, and robustness?`  
Please refer to the "Timeline" section where I have described in details about my plans to tackle different aspects of the project.  

`5. Does your project have any milestones that you can target throughout the period?`  
Yes, I would have implemented variable order and variable time step size multistep methods and IMEX methods.

`6. Are there any stretch goals you can make if the main project goes smoothly?`  
Yes, I would try to implement variable time step variable order BDF and SBDF methods.  

`7. What other time commitments, such as summer courses, other jobs, planned vacations, etc., will you have over the summer?`  
I expect to work full time on the project that is 40 or more hours a week.  

# Timeline (tentative)  
#### Community Bonding period (23rd April - 14th May)  
The primary focus in this period will be to have extensive discussions with my mentor on the implementation of Variable step size and variable order variable step size methods and modification in conditions of changing order and step size as it can be modified to achieve better result and implementation of IMEX methods.

#### Week 1 and 2
##### Goal :
* Implement Variable step size multistep methods.
* Add test cases and documentation

#### Week 3 and 4
##### Goal : 
* Implement variable order and variable step size multistep methods
* Add test cases and documentation

#### Week 5 and 6
##### Goal : 
* Implement ABCN, CNLF and Adaptive Step Size Second Order BDF method  
* Add test cases and documentation  

#### Week 7 and 8
##### Goal : 
* Implement second, third and fourth order SBDF (IMEX multistep methods)  
* Add test cases and documentation

#### Week 9 and 10
##### Goal : 
* Implement IMEX Runge-Kutta Methods
* Add test cases and documentation

#### Week 11 and 12
#### Goal : 
* The last two weeks would be a buffer period. In case some part of the project gets delayed, this period would be used for completing the project.  
* I also aim to work towards implementing the stretch goals.

## Any Plan/Commitment (During GSoC)
I have no major plans for summer and I will contribute full time for 40 or more hours a week. My college restarts in mid-July but I will still be able to contribute full time since there will be no exams or tests.

# References
[1] E. Hairer, S.R Nørsett and G. Wanner, Solving Ordinary Differential Equations I: Nonstiff Problems (Springer, New York, 1993).  
[2] E. Hairer and G. Wanner, Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems (Springer, New York, 1991).  
[3] [Implicit-Explicit Methods for ODEs](math.utah.edu/~vshankar/5620/IMEX.pdf)  
[4] [Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations](http://chaosbook.org/library/AscherANM97.pdf)  
[5] [IMPLICIT-EXPLICIT METHODS FOR TIME-DEPENDENT PDE'S](https://www.cs.ubc.ca/cgi-bin/tr/1993/TR-93-15.pdf)  
[6] https://github.com/JuliaDiffEq/ODE.jl/pull/106  
[7] [Implementation of an Adaptive BDF2 Formula and
Comparison with the MATLAB Ode15s](https://www.sciencedirect.com/science/article/pii/S1877050914002683)  
[8] [VARIABLE STEP-SIZE IMPLICIT-EXPLICIT LINEAR MULTISTEP
METHODS FOR TIME-DEPENDENT PARTIAL DIFFERENTIAL
EQUATIONS](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.204.5870&rep=rep1&type=pdf)
