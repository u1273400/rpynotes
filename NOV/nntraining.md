
## Cross validation

Cross validation is a method of splitting all your data into to parts: training and validation.  The training data is used to build the machine learning model, whereas the validation data is used to validate that the model is doing what is expected.  This increases our ability to find and determine the underlying errors in a model.  In a supervised environment, without an initial set to train on the algorithm would be useless.

Swapping training with validation helps increase the number of tests.  This id done by splitting the data into two; the first time set 1 is used to train and set two to validate.  The next step would be to swap the roles of set one and two for the subsequent test. Depending on how much data is available, the data is plsit into smaller sets and cross-validated. Given sufficient data, an indefinite amount of cross-validation sets can be setup.


## Training Algorithms

Common algorithms for training neural networks include the following.

1. Back propagation
2. Quick propagation
3. RProp

All of these algorithms find optimal weights for each neuron.  They do so through iterations known as epochs.  For each epoch, a training algorithm goes through the entire Neural Network and compares it against what is expected.  At this point, it learns from past miscalculations.

These algorithms have one thing in common: they are trying to find the optimal solution in a convex error surface.  You can think of convex error surface as a bowl with a minimum value in it.  These algorithms known as Gradient Descent minimize the error by using local information.

### The delta rule
This rule simplifies the aim of Gradient descent through iteration than heavy algebraic equations.  Instead of calculating the derivative error function with respect to the weight, we calculate the change in weight for each neuron's weights.  This delta rule is as follows:
$$\delta w_{ji}=\alpha(t_j-\phi(h_j))\phi'(h_j)x_i$$

This states that the change in weight for the neuron j's weight number i is:
> alpha \* (expected - calculated) \* derivative_of_calculated \*  input_at_i

alpha is the learning rate and is a small constant.  The initial idea, through is what powers the idea behind back propagation algorithm, or the general case of the delta rule.

### Back Propagation
Back Propagation is the simplest of the three algorithms that determine the weight of a neuron.  The error is defined as $(expected * actual)^2$ where expeceted is the expected output and actual is the calculated number from the neurons. We want to find where the derivative of this value is 0, which is the minimum:
$$\Delta w(t)=-\alpha(t-y)\phi'x_i+\epsilon\Delta w(t-1)$$

$\epsilon$ is the momentum factor and propels previous weight changes into our current weight change, whereas $\alpha$ is the learning rate.

Back propagation has the disadavantage of taking many epochs to calculate.  Up until 1988, researchers were struggling to train simple Neural Networks.  Their research on how to improve this led to a new algorithm called QuickProp.

### QuickProp
Scott Fahlman introduced the QuickProp algorithm after he studied how to improve Back Propagation.  He asserted that Back Propagation took  too long to converge to a solution.  He proposed that we instead take the biggest steps without overstepping the solution.

Fahlman determined that there are two ways to improve Back Propagations: making the momentum and learning rate dynamic, and making use of a second derivative of the error with respect to each weight.  In the first case, the algorithm can be optimized for each weight, and in the second case, Newton's method can be utilized to approximate functions.

With QuickProp, the main difference from Back Propagation is that you keep a copy of the error derivative computed during the previous epoch, along with the difference between the current an previous values of this weight.

To calculate a weight change at time t, the following function can be employed:

$$\Delta w(t)=\frac{S(t)}{S(t-1)-S(t)}\Delta w(t-1)$$

This carries the risk of changing the weights too much, so there is a new parameter for maximum growth.  No weight is allowed to be greater in magnitude than the maximum growth rate multiplied by the previous step for that weight.

### RProp

RProp is the most used algorithm because it converges fast.  It was introduced by Martin Riedmiller in the 1990s and has had improvements since then.  It converges on a solution quickly due to its insight that the algorithm can update the weights many times through an epoch.  Instead of calculating weight changes based on a formula, it uses only the sign for change as well as an increase factor and decrease factor.

To see what this algorithm looks like in code, defaults need to be defined. FAAN library has such defaults defined and based on the delta rule the following can be code utilized.


```ruby
neurons=3
inputs =4

delta_zero=0.1
increase_factor=1.2
decrease_factor=0.5
delta_max = 50.0
delta_min=0.0
max_epoch = 100
deltas = Array.new(inputs){Array.new(neurons){delta_zero}}
last_gradient = Array.new(inputs){Array.new(neurons){0.0}}
current_gradient = Array.new(inputs){Array.new(neurons){0.0}}

sign= -> (x){
  if x > 0
    1
  elsif x < 0
    -1
  else
    0
  end
}

weights = inputs.times.map{|i| rand(-1.0..1.0)}

1.upto(max_epoch)do |j|
  weights.each_with_index do |i,weight|
    # Current gradient is derived from the change of each value at each layer
    gradient_momentum=last_gradient[i][j] * current_gradient[i][j]
    
    if gradient_momentum > 0
      deltas[i][j]=[deltas[i][j] * increase_factor, delta_max].min
      change_weight=-sign.(current_gradient[i][j]) * delta[i][j]
      weights[i]=weight+change_weight
      last_gradient[i][j]=current_gradient[i]
    elsif gradient_momentum < 0
      deltas[i][j]=[deltas[i][j] * decrease_factor,delta_min].max
      last_gradient[i][j]=0
    else
      change_weight=-sign.(current_gradient[i][j]) * deltas[i][j]
      weights[i]=weights[i]+change_weight
      last_gradient[i][j]=current_gradient[i][j]
    end
  end
end
```


    NoMethodError: undefined method `*' for nil:NilClass

    (pry):207:in `block (2 levels) in <main>'

    (pry):205:in `each'

    (pry):205:in `each_with_index'

    (pry):205:in `block in <main>'

    (pry):204:in `upto'

    (pry):204:in `<main>'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:355:in `eval'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:355:in `evaluate_ruby'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:323:in `handle_line'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:243:in `block (2 levels) in eval'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:242:in `catch'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:242:in `block in eval'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:241:in `catch'

    /var/lib/gems/2.1.0/gems/pry-0.10.1/lib/pry/pry_instance.rb:241:in `eval'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/backend.rb:65:in `eval'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/backend.rb:12:in `eval'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/kernel.rb:87:in `execute_request'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/kernel.rb:47:in `dispatch'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/kernel.rb:37:in `run'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/command.rb:70:in `run_kernel'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/lib/iruby/command.rb:34:in `run'

    /var/lib/gems/2.1.0/gems/iruby-0.2.7/bin/iruby:5:in `<top (required)>'

    /usr/local/bin/iruby:23:in `load'

    /usr/local/bin/iruby:23:in `<main>'


### Training code files

1. getdata??
2. test/lib/language_spec.rb
3. lib/language.rb
4. lib/tokenizer.rb
5. test/cross_validation_spec.rb
6. lib/network.rb


```ruby

```
