<h1 id="neural-networks-explained-in-great-detail">Neural Networks Explained in Great Detail</h1>
<h2 id="introduction">Introduction</h2>
<p>Neural networks are a class of machine learning models inspired by the structure and functioning of the human brain. They are powerful algorithms capable of learning complex patterns and relationships in data. To understand neural networks better, let’s break them down into their individual parts and then explore how these components work together to create a functional network.</p>
<h2 id="the-neuron---building-block-of-neural-networks">The Neuron - Building Block of Neural Networks</h2>
<p>At the core of a neural network is the <strong>neuron</strong>. Just like the neurons in our brain, artificial neurons are responsible for processing and transmitting information. Each neuron takes multiple inputs, applies weights to those inputs, sums them up, and passes the result through an activation function to produce an output.</p>
<h3 id="math-behind-a-neuron">Math Behind a Neuron</h3>
<p>Mathematically, the operation of a neuron can be described as follows:</p>
<pre><code>output = activation_function(sum(input_values * weights) + bias)
</code></pre>
<p>Where:</p>
<ul>
<li><code>input_values</code>: The inputs coming from the previous layer or input data.</li>
<li><code>weights</code>: Corresponding weights for each input, indicating their importance.</li>
<li><code>bias</code>: A constant term to allow the model to learn an offset.</li>
</ul>
<h2 id="layers---organizing-neurons">Layers - Organizing Neurons</h2>
<p>A single neuron has limited capabilities, but when we combine multiple neurons, we form <strong>layers</strong>. Layers in a neural network organize neurons into groups, allowing for more complex computations.</p>
<h3 id="input-layer">Input Layer</h3>
<p>The first layer is called the <strong>input layer</strong>, which receives the raw input data. Each neuron in the input layer corresponds to a feature in the input data. For example, in an image recognition task, each pixel could be represented by an input neuron.</p>
<h3 id="hidden-layers">Hidden Layers</h3>
<p>Between the input and output layers, we can have one or more <strong>hidden layers</strong>. These layers are responsible for extracting meaningful patterns from the input data. Each neuron in a hidden layer takes inputs from the previous layer, performs its computations, and passes the results to the next layer.</p>
<h3 id="output-layer">Output Layer</h3>
<p>Finally, we have the <strong>output layer</strong>, which produces the final predictions or outputs of the neural network. The number of neurons in the output layer depends on the type of problem we are trying to solve. For instance, in a binary classification task, there would be one output neuron, while a multi-class classification task may require multiple output neurons.</p>
<h2 id="activation-functions---introducing-non-linearity">Activation Functions - Introducing Non-Linearity</h2>
<p>The activation function is a crucial component of a neuron that introduces non-linearity into the neural network. Without non-linear activation functions, the entire network would behave like a linear model, severely limiting its expressive power. This is because neural networks aim to find a relationship between some input and the output and most relationships are non-linear in nature.</p>
<h3 id="sigmoid-activation">Sigmoid Activation</h3>
<p>One of the earliest activation functions used is the <strong>sigmoid function</strong>. It maps the neuron’s input to a value between 0 and 1, which can represent probabilities. However, it suffers from the vanishing gradient problem, making it less suitable for deep neural networks. Equations and more explanation for this is below.</p>
<h3 id="relu-activation">ReLU Activation</h3>
<p>The <strong>Rectified Linear Unit (ReLU)</strong> activation overcomes the vanishing gradient problem and is widely used today. It maps all negative inputs to 0 and keeps positive inputs unchanged, adding simplicity and efficiency to the network.</p>
<h3 id="other-activation-functions">Other Activation Functions</h3>
<p>There are various other activation functions like <strong>tanh</strong>, <strong>Leaky ReLU</strong>, and <strong>softmax</strong> (for multi-class classification), each with its strengths and use cases.</p>
<h2 id="forward-propagation---feeding-data-through-the-network">Forward Propagation - Feeding Data through the Network</h2>
<p>To compute the output of a neural network, we use <strong>forward propagation</strong>. The data flows from the input layer, through the hidden layers, and finally to the output layer. Each neuron’s output is computed as discussed earlier, using the activation function.</p>
<h2 id="loss-function---measuring-the-error">Loss Function - Measuring the Error</h2>
<p>The <strong>loss function</strong> evaluates how well the neural network’s predictions match the actual targets (ground truth). The choice of loss function depends on the problem type: regression, classification, etc. The goal is to minimize the loss during training.</p>
<h2 id="backpropagation---learning-from-mistakes">Backpropagation - Learning from Mistakes</h2>
<p>The process of adjusting the network’s weights and biases based on the loss function’s output is known as <strong>backpropagation</strong>. It is the key to training neural networks. Backpropagation calculates the gradients of the loss with respect to each weight and bias and updates them in the opposite direction to minimize the loss. Don’t worry. This is also explained in greater detail below.</p>
<h2 id="training---iterative-learning">Training - Iterative Learning</h2>
<p>Training a neural network involves iteratively feeding the input data, computing the output through forward propagation, calculating the loss, and then updating the model’s parameters using backpropagation. This process continues for multiple epochs (iterations) until the network converges and the loss is minimized.</p>
<h2 id="conclusion">Conclusion</h2>
<p>Neural networks are complex and powerful models that can learn and generalize from data. They consist of individual neurons organized into layers, where each neuron applies weights and an activation function to process inputs. Training neural networks involves adjusting these weights and biases using backpropagation to minimize the loss. By stacking multiple layers and using non-linear activation functions, neural networks can tackle a wide range of tasks, from image recognition to natural language processing.</p>
<h2 id="the-neuron-a-more-detailed-explanation">The Neuron: A More Detailed Explanation</h2>
<h3 id="introduction-1">Introduction</h3>
<p>A neuron is a fundamental unit of a neural network, responsible for processing and transmitting information. It is inspired by the structure and function of biological neurons found in the human brain. Understanding the workings of a single neuron is essential to comprehend the overall functioning of neural networks.</p>
<h3 id="anatomy-of-a-neuron">Anatomy of a Neuron</h3>
<p>A typical artificial neuron can be divided into three main parts:</p>
<ol>
<li><strong>Dendrites</strong>: These are like the input arms of the neuron, collecting signals from other neurons or the external environment.</li>
<li><strong>Cell Body (Soma)</strong>: The cell body acts as the processing center, where the input signals get integrated and modified.</li>
<li><strong>Axon</strong>: The axon is like the output wire of the neuron, transmitting the processed signal to other connected neurons or the final output.</li>
</ol>
<h3 id="analogy-the-postal-worker">Analogy: The Postal Worker</h3>
<p>To better understand how a neuron works, let’s use an analogy involving a postal worker. Imagine a postal worker (the neuron) situated at a junction in a small town.</p>
<ol>
<li>
<p><strong>Dendrites</strong>: The postal worker receives letters (input signals) from multiple mailboxes (other neurons) in the neighborhood. Each mailbox represents an input, and the number written on the letter signifies the strength (weight) of the signal.</p>
</li>
<li>
<p><strong>Cell Body (Soma)</strong>: The postal worker collects all the letters and takes them to the sorting room (cell body). Inside the sorting room, the postal worker adds up the values written on each letter, reflecting the neuron’s weighted sum of inputs.</p>
</li>
<li>
<p><strong>Axon</strong>: After summing up the values, the postal worker checks whether the total sum of values exceeds a certain threshold (bias). If it does, the postal worker sends a new letter (output signal) to the next junction or directly to the destination mailbox (output neuron). This letter carries information about the outcome of the neuron’s computation.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-a-neuron">Mathematical Representation of a Neuron</h3>
<p>The computation performed by a neuron can be mathematically expressed as follows:</p>
<pre><code>input_values = [x₁, x₂, x₃, ..., xₙ]  # Inputs received from other neurons or input data
weights = [w₁, w₂, w₃, ..., wₙ]      # Corresponding weights for each input
bias = b                              # A constant term to allow the neuron to learn an offset

weighted_sum = Σ (input_values * weights) + bias
output = activation_function(weighted_sum)
</code></pre>
<p>Where:</p>
<ul>
<li><code>input_values</code>: Array of inputs received by the neuron.</li>
<li><code>weights</code>: Array of corresponding weights for each input, indicating their importance.</li>
<li><code>bias</code>: A constant term, which can be thought of as a threshold for activation.</li>
<li><code>weighted_sum</code>: The sum of the element-wise multiplication of inputs and weights, combined with the bias term.</li>
<li><code>activation_function</code>: The activation function applied to the weighted sum, producing the neuron’s output.</li>
</ul>
<p>The output of the neuron is the result of applying the activation function to the weighted sum. The choice of activation function introduces non-linearity, enabling neural networks to model complex relationships in data.</p>
<p>Basically, a neuron is like a postal worker; it receives information, processes it, and then forwards the result to other neurons or as the final output, playing a crucial role in the information flow and decision-making within a neural network.</p>
<h2 id="the-layer-a-more-detailed-explanation">The Layer: A More Detailed Explanation</h2>
<h3 id="introduction-2">Introduction</h3>
<p>In a neural network, a layer is a collection of neurons that work together to process and transform data. Layers play a crucial role in organizing and orchestrating the flow of information throughout the network. Understanding the concept of a layer is essential to grasp the overall functioning of neural networks.</p>
<h3 id="anatomy-of-a-layer">Anatomy of a Layer</h3>
<p>A layer consists of multiple neurons arranged in a specific pattern. Each neuron in a layer receives input from the previous layer (or the input data, in the case of the input layer), processes that input independently, and produces an output. The outputs from all neurons in the layer collectively form the output of the entire layer.</p>
<h3 id="analogy-the-team-of-specialists">Analogy: The Team of Specialists</h3>
<p>To better understand how a layer functions, let’s use an analogy involving a team of specialists working on a complex project.</p>
<p>Imagine a company working on designing a cutting-edge product, and they have assembled a team of specialists to handle different aspects of the project.</p>
<ol>
<li>
<p><strong>Input</strong>: The team leader (the neural network) receives the project requirements (input data) from the client (external environment) and shares it with the team.</p>
</li>
<li>
<p><strong>Layer</strong>: Each specialist (neuron) in the team (layer) focuses on a specific part of the project (processing certain aspects of the input data). For example, one specialist may handle the design, another the electronics, and another the materials.</p>
</li>
<li>
<p><strong>Output</strong>: After processing their respective parts, each specialist presents their findings (neuron outputs) to the team leader. The team leader combines all the outputs to get a complete understanding of the project’s progress (layer output).</p>
</li>
</ol>
<h3 id="mathematical-representation-of-a-layer">Mathematical Representation of a Layer</h3>
<p>Mathematically, the computation performed by a layer can be represented as follows:</p>
<pre><code>input_values = [x₁, x₂, x₃, ..., xₙ]  # Inputs received from the previous layer or input data
weights = [
    [w₁₁, w₁₂, w₁₃, ..., w₁ₙ],  # Weights connecting input to neurons in the layer
    [w₂₁, w₂₂, w₂₃, ..., w₂ₙ],
    ...
    [wₖ₁, wₖ₂, wₖ₃, ..., wₖₙ],
]
bias = [b₁, b₂, ..., bₖ]  # Biases for each neuron in the layer

output = [
    activation_function(Σ (input_values * weights[1]) + b₁),  # Output of neuron 1
    activation_function(Σ (input_values * weights[2]) + b₂),  # Output of neuron 2
    ...
    activation_function(Σ (input_values * weights[𝑘]) + bₖ),  # Output of neuron 𝑘
]
</code></pre>
<p>Where:</p>
<ul>
<li><code>input_values</code>: Array of inputs received from the previous layer or input data.</li>
<li><code>weights</code>: A matrix of weights that connect the input to each neuron in the layer.</li>
<li><code>bias</code>: An array of biases, one for each neuron, to allow them to learn an offset.</li>
<li><code>activation_function</code>: The activation function applied to each neuron’s weighted sum, producing the neuron’s output.</li>
<li><code>output</code>: An array containing the output of each neuron in the layer.</li>
</ul>
<p>Soooo, a layer, like a team of specialists, processes and transforms the input data independently, contributing to the overall progress of the project (the neural network’s computation). By stacking multiple layers together, a neural network gains the ability to learn complex patterns and relationships in the data, much like the diverse expertise of a team can lead to groundbreaking innovations.</p>
<h2 id="activation-functions-a-more-detailed-explanation">Activation Functions: A More Detailed Explanation</h2>
<h3 id="introduction-3">Introduction</h3>
<p>An activation function is a crucial component of a neuron in a neural network. It introduces non-linearity to the model, enabling it to learn and approximate complex relationships in the data. Activation functions determine whether a neuron “fires” or gets activated based on the input it receives. Understanding the role of activation functions is essential to grasp the power of neural networks.</p>
<h3 id="the-role-of-activation-functions">The Role of Activation Functions</h3>
<p>An activation function takes the weighted sum of inputs in a neuron and transforms it into an output value. This output value serves as the neuron’s actual output, which is passed on to the next layer or used to make predictions in the case of the output layer. Activation functions introduce non-linearities into the neural network, which allows it to learn more complex and sophisticated patterns in the data.</p>
<h3 id="analogy-the-light-switch">Analogy: The Light Switch</h3>
<p>To better understand the role of an activation function, let’s use an analogy involving a light switch.</p>
<p>Imagine a room with a light switch, and the light switch controls the brightness of a lamp. The lamp represents the output of the neuron, while the switch represents the activation function.</p>
<ol>
<li>
<p><strong>Linear Activation</strong>: If we use a linear activation function, it’s like having a simple on/off switch. No matter how much we tweak the switch (adjust the input values), the lamp’s brightness (output) will remain the same. There’s no room for subtlety or nuance in the brightness level.</p>
</li>
<li>
<p><strong>Non-linear Activation</strong>: Now, let’s replace the on/off switch with a dimmer switch. With a dimmer switch, we can precisely control the brightness of the lamp based on how much we move the switch. This allows for a wide range of brightness levels (non-linearity), adding complexity and richness to the room’s lighting.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-activation-functions">Mathematical Representation of Activation Functions</h3>
<p>Mathematically, an activation function takes the weighted sum of inputs (often denoted as <code>z</code>) and applies a non-linear transformation to it, producing the neuron’s output <code>a</code>.</p>
<pre><code>z = Σ (input_values * weights) + bias  # Weighted sum of inputs and biases
a = activation_function(z)            # Output of the neuron after applying the activation function
</code></pre>
<p>Commonly used activation functions include:</p>
<ol>
<li><strong>Sigmoid Activation</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python">sigmoid<span class="token punctuation">(</span>z<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token number">1</span> <span class="token operator">/</span> <span class="token punctuation">(</span><span class="token number">1</span> <span class="token operator">+</span> exp<span class="token punctuation">(</span><span class="token operator">-</span>z<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<ol start="2">
<li><strong>ReLU Activation</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python">ReLU<span class="token punctuation">(</span>z<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token builtin">max</span><span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span> z<span class="token punctuation">)</span>
</code></pre>
<ol start="3">
<li><strong>Tanh Activation</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python">tanh<span class="token punctuation">(</span>z<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token punctuation">(</span>exp<span class="token punctuation">(</span>z<span class="token punctuation">)</span> <span class="token operator">-</span> exp<span class="token punctuation">(</span><span class="token operator">-</span>z<span class="token punctuation">)</span><span class="token punctuation">)</span> <span class="token operator">/</span> <span class="token punctuation">(</span>exp<span class="token punctuation">(</span>z<span class="token punctuation">)</span> <span class="token operator">+</span> exp<span class="token punctuation">(</span><span class="token operator">-</span>z<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<h3 id="why-are-activation-functions-necessary">Why Are Activation Functions Necessary?</h3>
<p>Activation functions are essential in neural networks for several reasons:</p>
<ol>
<li>
<p><strong>Introducing Non-linearity</strong>: Without non-linear activation functions, the entire neural network would behave like a linear model. Non-linearity is crucial for modeling complex patterns and relationships in data.</p>
</li>
<li>
<p><strong>Expressive Power</strong>: Non-linear activation functions allow neural networks to approximate any continuous function, making them universal function approximators.</p>
</li>
<li>
<p><strong>Learning Complex Representations</strong>: The multi-layered structure of neural networks allows them to learn hierarchical representations of data, and activation functions enable the learning of intricate features at different layers.</p>
</li>
<li>
<p><strong>Mitigating the Vanishing Gradient Problem</strong>: Certain activation functions, like ReLU, help alleviate the vanishing gradient problem during training, leading to more stable and efficient learning.</p>
</li>
</ol>
<p>Activation functions act as “switches” that control the firing of neurons, introducing non-linearity to the neural network. This non-linearity allows the network to learn complex patterns, making it a powerful tool for solving various machine learning problems.</p>
<h2 id="the-vanishing-gradient-problem-and-activation-functions">The Vanishing Gradient Problem and Activation Functions</h2>
<h3 id="the-vanishing-gradient-problem">The Vanishing Gradient Problem</h3>
<p>The vanishing gradient problem is a challenge that arises during the training of deep neural networks, particularly those with many layers. It occurs when the gradients of the loss function with respect to the model’s parameters (weights and biases) become extremely small as they propagate backward through the network during the process of backpropagation.</p>
<p>During backpropagation, the gradients are used to update the model’s parameters to minimize the loss function, effectively optimizing the network for better performance. However, if the gradients become too small, the updates to the parameters become insignificant, leading to slow or stalled learning.</p>
<p>The vanishing gradient problem is most evident in networks that use activation functions with small gradients in certain input ranges, such as the sigmoid activation function.</p>
<h3 id="mitigation-with-activation-functions">Mitigation with Activation Functions</h3>
<p>Activation functions play a crucial role in mitigating the vanishing gradient problem. Some activation functions are better suited for deep neural networks because they exhibit better gradient behavior, enabling smoother and more efficient learning. Let’s explore how some activation functions help mitigate the vanishing gradient problem:</p>
<ol>
<li>
<p><strong>ReLU (Rectified Linear Unit)</strong>:</p>
<ul>
<li>ReLU has a simple form: <code>ReLU(z) = max(0, z)</code>.</li>
<li>It is computationally efficient and has a derivative of 1 for positive inputs and 0 for negative inputs.</li>
<li>The derivative being 1 for positive inputs avoids the vanishing gradient problem in the forward pass, as there is no shrinking effect on positive gradients.</li>
</ul>
</li>
<li>
<p><strong>Leaky ReLU</strong>:</p>
<ul>
<li>Leaky ReLU is a variation of ReLU that addresses the “dying ReLU” problem, where neurons can get stuck during training and never activate again.</li>
<li>Leaky ReLU introduces a small positive slope for negative inputs, which ensures that there is a non-zero gradient even for negative values.</li>
</ul>
</li>
<li>
<p><strong>Parametric ReLU (PReLU)</strong>:</p>
<ul>
<li>PReLU is a generalization of Leaky ReLU, where the slope for negative inputs is learned during training.</li>
<li>This adaptability helps the network find the most suitable slope, reducing the likelihood of the vanishing gradient problem.</li>
</ul>
</li>
<li>
<p><strong>ELU (Exponential Linear Unit)</strong>:</p>
<ul>
<li>ELU is similar to Leaky ReLU but with a smooth exponential curve for negative inputs.</li>
<li>The smoothness of the function allows it to capture more information and mitigate the vanishing gradient problem.</li>
</ul>
</li>
</ol>
<p>These activation functions, especially ReLU and its variants, have become popular choices for deep neural networks because they alleviate the vanishing gradient problem, allowing for faster and more effective learning in deep architectures. By ensuring non-zero gradients in specific input ranges, these activation functions help maintain the flow of information during backpropagation, enabling the network to learn meaningful representations and perform better on complex tasks.</p>
<h2 id="forward-propagation-a-more-detailed-explanation">Forward Propagation: A More Detailed Explanation</h2>
<h3 id="introduction-4">Introduction</h3>
<p>Forward propagation is a fundamental process in a neural network, where the input data flows through the network from the input layer to the output layer. During forward propagation, each neuron in the network receives the input from the previous layer (or directly from the input data) and performs its computations to produce an output. Understanding forward propagation is essential to comprehend how neural networks make predictions.</p>
<h3 id="the-information-flow">The Information Flow</h3>
<p>Forward propagation can be visualized as passing a message through a chain of people, where each person represents a neuron in the network.</p>
<p>Imagine a relay race with a team of runners. The race is divided into multiple sections, and each runner carries a baton (the information) from one section to the next. The first runner starts the race by receiving the baton (input data) and runs through their section (neuron computations). They then pass the baton to the next runner (the next layer) who continues the race in the same way.</p>
<h3 id="mathematical-representation-of-forward-propagation">Mathematical Representation of Forward Propagation</h3>
<p>Mathematically, forward propagation involves the following steps:</p>
<ol>
<li>
<p><strong>Input Data</strong>: The input data is provided to the neural network, and each neuron in the input layer receives a specific feature from the data.</p>
</li>
<li>
<p><strong>Weighted Sum</strong>: Each neuron calculates the weighted sum of its inputs, including the biases. This is equivalent to each runner running through their section of the race and summing up their progress.</p>
</li>
<li>
<p><strong>Activation Function</strong>: The weighted sum is passed through an activation function, producing the neuron’s output. This represents each runner completing their section of the race and handing over the baton to the next runner.</p>
</li>
<li>
<p><strong>Propagation to Next Layer</strong>: The output of each neuron becomes the input for the neurons in the next layer, and the process repeats for each layer until the final output layer is reached. This is akin to each runner successfully passing the baton to the next runner in the relay race until the race is completed.</p>
</li>
</ol>
<h3 id="mathematical-representation-in-code-block">Mathematical Representation in Code Block</h3>
<p>Let’s represent the forward propagation process mathematically using code blocks:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Forward propagation for a single neuron in a layer</span>
<span class="token keyword">def</span> <span class="token function">forward_propagation_single_neuron</span><span class="token punctuation">(</span>inputs<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> bias<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    weighted_sum <span class="token operator">=</span> <span class="token builtin">sum</span><span class="token punctuation">(</span>inputs <span class="token operator">*</span> weights<span class="token punctuation">)</span> <span class="token operator">+</span> bias
    output <span class="token operator">=</span> activation_function<span class="token punctuation">(</span>weighted_sum<span class="token punctuation">)</span>
    <span class="token keyword">return</span> output

<span class="token comment"># Forward propagation for a layer</span>
<span class="token keyword">def</span> <span class="token function">forward_propagation_layer</span><span class="token punctuation">(</span>input_values<span class="token punctuation">,</span> weights_matrix<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    outputs <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>weights_matrix<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        neuron_output <span class="token operator">=</span> forward_propagation_single_neuron<span class="token punctuation">(</span>input_values<span class="token punctuation">,</span> weights_matrix<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> biases<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>
        outputs<span class="token punctuation">.</span>append<span class="token punctuation">(</span>neuron_output<span class="token punctuation">)</span>
    <span class="token keyword">return</span> outputs

<span class="token comment"># Complete forward propagation for the entire neural network</span>
<span class="token keyword">def</span> <span class="token function">forward_propagation_neural_network</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    current_layer_inputs <span class="token operator">=</span> input_data
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        current_layer_outputs <span class="token operator">=</span> forward_propagation_layer<span class="token punctuation">(</span>current_layer_inputs<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> biases<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>
        current_layer_inputs <span class="token operator">=</span> current_layer_outputs
    <span class="token keyword">return</span> current_layer_outputs  <span class="token comment"># The final output of the neural network</span>
</code></pre>
<p>And that is it. Forward propagation is like a relay race where the input data flows through the network, passing through each neuron’s computations and activation functions until the final output is produced. It allows the neural network to make predictions and learn meaningful representations from the data it receives.</p>
<h2 id="loss-functions-a-more-detailed-explanation">Loss Functions: A More Detailed Explanation</h2>
<h3 id="introduction-5">Introduction</h3>
<p>A loss function, also known as a cost function or objective function, is a crucial component of a machine learning model, including neural networks. It measures how well the model’s predictions match the actual target values. The goal of training a model is to minimize the loss function, which guides the model to improve its performance over time.</p>
<h3 id="the-analogy-a-marks-grading-system">The Analogy: A Marks Grading System</h3>
<p>To better understand the concept of a loss function, let’s use an analogy involving a marks grading system in a classroom.</p>
<p>Imagine you are a teacher, and your students have just taken a test. You want to evaluate how well each student performed on the test. To do this, you create a grading system (the loss function) that assesses the difference between the students’ actual marks (target values) and their predicted marks (model’s predictions).</p>
<h3 id="the-loss-function-in-detail">The Loss Function in Detail</h3>
<ol>
<li>
<p><strong>Collecting Data</strong>: You collect the test papers with the students’ answers and their corresponding correct answers. Each student’s score on the test represents the actual marks (target values).</p>
</li>
<li>
<p><strong>Model Prediction</strong>: To evaluate the students’ performance, you ask another teacher (the model) to grade the test papers. The other teacher provides predicted scores for each student.</p>
</li>
<li>
<p><strong>Loss Calculation</strong>: Now, you need to compare the predicted scores with the actual scores to assess the model’s accuracy. The loss function quantifies this difference between the predicted and actual scores.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-a-loss-function">Mathematical Representation of a Loss Function</h3>
<p>Mathematically, a loss function is typically denoted by <code>L(y_true, y_pred)</code>, where <code>y_true</code> represents the actual target values (ground truth), and <code>y_pred</code> represents the model’s predicted values.</p>
<p>Here are some common loss functions used in machine learning and neural networks:</p>
<ol>
<li><strong>Mean Squared Error (MSE)</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python">MSE<span class="token punctuation">(</span>y_true<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">/</span>n<span class="token punctuation">)</span> <span class="token operator">*</span> Σ<span class="token punctuation">(</span>y_true <span class="token operator">-</span> y_pred<span class="token punctuation">)</span><span class="token operator">^</span><span class="token number">2</span>
</code></pre>
<ol start="2">
<li><strong>Binary Cross-Entropy (Log Loss)</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python">Binary_CE<span class="token punctuation">(</span>y_true<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token operator">-</span> <span class="token punctuation">(</span>y_true <span class="token operator">*</span> log<span class="token punctuation">(</span>y_pred<span class="token punctuation">)</span> <span class="token operator">+</span> <span class="token punctuation">(</span><span class="token number">1</span> <span class="token operator">-</span> y_true<span class="token punctuation">)</span> <span class="token operator">*</span> log<span class="token punctuation">(</span><span class="token number">1</span> <span class="token operator">-</span> y_pred<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<ol start="3">
<li><strong>Categorical Cross-Entropy (Multi-Class Cross-Entropy)</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python">Categorical_CE<span class="token punctuation">(</span>y_true<span class="token punctuation">,</span> y_pred<span class="token punctuation">)</span> <span class="token operator">=</span> <span class="token operator">-</span> Σ<span class="token punctuation">(</span>y_true <span class="token operator">*</span> log<span class="token punctuation">(</span>y_pred<span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<h3 id="loss-minimization">Loss Minimization</h3>
<p>The objective during model training is to minimize the value of the loss function. Minimizing the loss means making the model’s predictions as close to the actual target values as possible. This process is typically achieved using optimization algorithms like Gradient Descent, which iteratively update the model’s parameters to find the values that lead to the smallest loss.</p>
<p>A loss function acts as a grading system for machine learning models. It quantifies the discrepancy between the model’s predictions and the actual target values. By minimizing the loss function, the model learns to make better predictions and improve its performance on various tasks. Just as a teacher seeks to grade students’ performance accurately, a well-chosen loss function guides a machine learning model to learn from its mistakes and make more accurate predictions.</p>
<h2 id="backpropagation-a-more-detailed-explanation">Backpropagation: A More Detailed Explanation</h2>
<h3 id="introduction-6">Introduction</h3>
<p>Backpropagation is a fundamental algorithm used for training neural networks. It is responsible for updating the model’s parameters (weights and biases) based on the calculated gradients of the loss function with respect to those parameters. Backpropagation allows the network to learn from its mistakes and adjust its parameters to improve performance.</p>
<h3 id="the-analogy-the-gps-navigation-system">The Analogy: The GPS Navigation System</h3>
<p>To better understand the concept of backpropagation, let’s use an analogy involving a GPS navigation system.</p>
<p>Imagine you are driving a car, and you input your destination into the GPS navigation system. The GPS starts guiding you along a route, but initially, it might not find the shortest or most efficient path. However, as you follow the GPS directions and drive, the system continuously recalculates and updates the route based on your real-time position and the distance from the destination.</p>
<h3 id="backpropagation-in-detail">Backpropagation in Detail</h3>
<ol>
<li>
<p><strong>Initial Predictions</strong>: At the start of the journey, the GPS navigation system provides initial directions based on the map (the neural network’s initial parameters).</p>
</li>
<li>
<p><strong>Loss Calculation</strong>: As you drive, the GPS system continuously calculates the difference between your actual position and the desired destination. This discrepancy represents the loss (error) in the GPS’s directions.</p>
</li>
<li>
<p><strong>Updating Directions</strong>: The GPS system uses this loss information to adjust its directions for the next steps. It updates the route, aiming to minimize the difference between your actual position and the desired destination (minimizing the loss).</p>
</li>
<li>
<p><strong>Iterative Process</strong>: As you continue driving, the GPS recalculates and updates the route after each step, gradually providing more accurate directions based on real-time feedback.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-backpropagation">Mathematical Representation of Backpropagation</h3>
<p>In a neural network, backpropagation is the process of calculating gradients and using them to update the model’s parameters. Mathematically, it involves the following steps:</p>
<ol>
<li>
<p><strong>Forward Propagation</strong>: During forward propagation, the input data passes through the network, and the model computes predictions.</p>
</li>
<li>
<p><strong>Loss Calculation</strong>: The loss function evaluates how well the model’s predictions match the actual target values.</p>
</li>
<li>
<p><strong>Backward Pass</strong>: The gradients of the loss function with respect to the model’s parameters are computed using the chain rule of calculus.</p>
</li>
<li>
<p><strong>Parameter Updates</strong>: The model’s parameters are updated using the computed gradients and an optimization algorithm (e.g., Gradient Descent).</p>
</li>
</ol>
<h3 id="mathematical-representation-in-code-block-1">Mathematical Representation in Code Block</h3>
<p>Let’s represent the backpropagation process mathematically using code blocks:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Step 1: Forward Propagation</span>
<span class="token keyword">def</span> <span class="token function">forward_propagation</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    current_layer_inputs <span class="token operator">=</span> input_data
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        current_layer_outputs <span class="token operator">=</span> forward_propagation_layer<span class="token punctuation">(</span>current_layer_inputs<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> biases<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>
        current_layer_inputs <span class="token operator">=</span> current_layer_outputs
    <span class="token keyword">return</span> current_layer_outputs  <span class="token comment"># The final output of the neural network</span>

<span class="token comment"># Step 2: Loss Calculation (assuming Mean Squared Error)</span>
<span class="token keyword">def</span> <span class="token function">calculate_loss</span><span class="token punctuation">(</span>predictions<span class="token punctuation">,</span> target_values<span class="token punctuation">)</span><span class="token punctuation">:</span>
    n <span class="token operator">=</span> <span class="token builtin">len</span><span class="token punctuation">(</span>predictions<span class="token punctuation">)</span>
    loss <span class="token operator">=</span> <span class="token punctuation">(</span><span class="token number">1</span><span class="token operator">/</span>n<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token builtin">sum</span><span class="token punctuation">(</span><span class="token punctuation">(</span>target_values <span class="token operator">-</span> predictions<span class="token punctuation">)</span> <span class="token operator">**</span> <span class="token number">2</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> loss

<span class="token comment"># Step 3: Backward Pass</span>
<span class="token keyword">def</span> <span class="token function">backward_propagation</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># Forward propagation to get predictions</span>
    predictions <span class="token operator">=</span> forward_propagation<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>

    <span class="token comment"># Calculate loss and gradients for the output layer</span>
    loss <span class="token operator">=</span> calculate_loss<span class="token punctuation">(</span>predictions<span class="token punctuation">,</span> target_values<span class="token punctuation">)</span>
    output_gradient <span class="token operator">=</span> <span class="token punctuation">(</span><span class="token number">2</span><span class="token operator">/</span>n<span class="token punctuation">)</span> <span class="token operator">*</span> <span class="token punctuation">(</span>predictions <span class="token operator">-</span> target_values<span class="token punctuation">)</span>
    output_gradients <span class="token operator">=</span> <span class="token punctuation">[</span>output_gradient<span class="token punctuation">]</span>

    <span class="token comment"># Backpropagate the gradients to previous layers</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">)</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> <span class="token number">0</span><span class="token punctuation">,</span> <span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        hidden_gradient <span class="token operator">=</span> output_gradients<span class="token punctuation">[</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">]</span> @ weights_matrices<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">.</span>T <span class="token operator">*</span> activation_derivative<span class="token punctuation">(</span>forward_propagation_layer<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> biases<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">)</span>
        output_gradients<span class="token punctuation">.</span>insert<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">,</span> hidden_gradient<span class="token punctuation">)</span>

    <span class="token keyword">return</span> output_gradients

<span class="token comment"># Step 4: Parameter Updates (using Gradient Descent)</span>
<span class="token keyword">def</span> <span class="token function">update_parameters</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> gradients<span class="token punctuation">,</span> learning_rate<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        weights_matrices<span class="token punctuation">[</span>i<span class="token punctuation">]</span> <span class="token operator">-=</span> learning_rate <span class="token operator">*</span> <span class="token punctuation">(</span>input_data<span class="token punctuation">.</span>T @ gradients<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">)</span>
        biases<span class="token punctuation">[</span>i<span class="token punctuation">]</span> <span class="token operator">-=</span> learning_rate <span class="token operator">*</span> <span class="token builtin">sum</span><span class="token punctuation">(</span>gradients<span class="token punctuation">[</span>i<span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<p>Backpropagation in neural networks is like a GPS navigation system continuously recalculating and updating directions based on the distance between your actual position and the desired destination. Similarly, backpropagation iteratively computes gradients and updates the model’s parameters based on the loss, guiding the neural network to make more accurate predictions and learn from its mistakes.</p>
<h2 id="training-in-neural-networks">Training in Neural Networks</h2>
<h3 id="introduction-7">Introduction</h3>
<p>Training a neural network is the process of teaching the model to make accurate predictions by adjusting its parameters (weights and biases) based on the provided data and the desired outcomes. The ultimate goal of training is to minimize the difference between the model’s predictions and the actual target values. This is achieved through an iterative optimization process using a combination of forward and backward propagation.</p>
<h3 id="the-analogy-a-cooking-recipe">The Analogy: A Cooking Recipe</h3>
<p>To better understand the training process of a neural network, let’s use an analogy involving a cooking recipe.</p>
<p>Imagine you are a chef trying to create a new dish. You have a basic recipe (the initial model architecture) but need to adjust the ingredients and cooking times (model parameters) to achieve the perfect taste (accurate predictions). To do this, you follow a tasting-feedback loop, adjusting the recipe based on the feedback (loss evaluation) from your taste testers (training data).</p>
<h3 id="training-in-detail">Training in Detail</h3>
<ol>
<li>
<p><strong>Gather Ingredients</strong>: You collect all the ingredients needed for the dish, representing the input data and target values for the neural network.</p>
</li>
<li>
<p><strong>Follow Recipe</strong>: Initially, you follow the recipe step by step, representing the forward propagation process where the input data flows through the network, and the model makes its initial predictions.</p>
</li>
<li>
<p><strong>Evaluate Taste</strong>: You serve the dish to your taste testers, and they provide feedback on the taste (loss evaluation). The feedback indicates how far the dish’s taste is from the ideal taste (desired predictions).</p>
</li>
<li>
<p><strong>Adjust Ingredients</strong>: Based on the feedback, you start adjusting the amount of each ingredient and cooking time, representing the backward propagation process. The goal is to bring the taste closer to perfection (minimize the loss).</p>
</li>
<li>
<p><strong>Taste Again</strong>: After making the adjustments, you serve the dish again and gather new feedback. You repeat this process iteratively, making small tweaks each time.</p>
</li>
<li>
<p><strong>Convergence</strong>: As you continue to adjust the ingredients and cooking times based on feedback, the dish’s taste steadily improves. Eventually, you reach a point where making further adjustments doesn’t significantly change the taste (convergence). At this point, your dish is perfectly cooked, and your neural network is well-trained.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-training">Mathematical Representation of Training</h3>
<p>The training process in neural networks involves forward propagation, loss calculation, backward propagation, and parameter updates. Let’s represent these steps mathematically using code blocks:</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Forward Propagation (defined earlier)</span>
<span class="token keyword">def</span> <span class="token function">forward_propagation</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for forward propagation)</span>

<span class="token comment"># Loss Calculation (defined earlier)</span>
<span class="token keyword">def</span> <span class="token function">calculate_loss</span><span class="token punctuation">(</span>predictions<span class="token punctuation">,</span> target_values<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for loss calculation)</span>

<span class="token comment"># Backward Propagation (defined earlier)</span>
<span class="token keyword">def</span> <span class="token function">backward_propagation</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for backward propagation)</span>

<span class="token comment"># Parameter Updates (defined earlier)</span>
<span class="token keyword">def</span> <span class="token function">update_parameters</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> gradients<span class="token punctuation">,</span> learning_rate<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for parameter updates)</span>

<span class="token comment"># Complete Training Loop</span>
<span class="token keyword">def</span> <span class="token function">train_neural_network</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">,</span> learning_rate<span class="token punctuation">,</span> epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
        <span class="token comment"># Forward propagation to get predictions</span>
        predictions <span class="token operator">=</span> forward_propagation<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>

        <span class="token comment"># Calculate loss</span>
        loss <span class="token operator">=</span> calculate_loss<span class="token punctuation">(</span>predictions<span class="token punctuation">,</span> target_values<span class="token punctuation">)</span>

        <span class="token comment"># Backward propagation to compute gradients</span>
        gradients <span class="token operator">=</span> backward_propagation<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">)</span>

        <span class="token comment"># Update model parameters</span>
        update_parameters<span class="token punctuation">(</span>weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> gradients<span class="token punctuation">,</span> learning_rate<span class="token punctuation">)</span>

    <span class="token keyword">return</span> weights_matrices<span class="token punctuation">,</span> biases
</code></pre>
<p>Training a neural network is like fine-tuning a cooking recipe based on feedback from taste testers. The iterative process of adjusting the ingredients (model parameters) using the feedback (loss evaluation) guides the model to make more accurate predictions over time. Just as a chef refines a dish to perfection, training a neural network involves iteratively optimizing the model’s parameters to achieve the best possible predictions.</p>
<h2 id="how-a-neural-network-works">How a Neural Network Works</h2>
<h3 id="introduction-8">Introduction</h3>
<p>A neural network is a powerful machine learning model inspired by the human brain’s neural structure. It consists of interconnected layers of neurons that work together to process input data and make predictions. The key to a neural network’s effectiveness lies in its ability to learn and adjust its parameters based on the provided data and the desired outcomes.</p>
<h3 id="the-analogy-the-orchestra-conductor">The Analogy: The Orchestra Conductor</h3>
<p>To understand how a neural network works, let’s use an analogy involving an orchestra conductor.</p>
<p>Imagine you are an orchestra conductor, and you want to guide your orchestra (the neural network) to perform a musical piece (solve a specific task). The musicians in the orchestra represent the neurons, and each musician plays a different instrument (specific role) in the piece.</p>
<h3 id="how-a-neural-network-works-in-detail">How a Neural Network Works in Detail</h3>
<ol>
<li>
<p><strong>The Score (Model Architecture)</strong>: As the conductor, you start with the musical score (the model architecture) that outlines how the orchestra will play the piece. The score specifies the arrangement of musicians (neurons) in different sections (layers) of the orchestra.</p>
</li>
<li>
<p><strong>Rehearsal (Training)</strong>: Before the actual performance, you need to rehearse the piece with your orchestra. During the rehearsal, you provide input data (notes on a sheet) and the desired musical outcome (the correct melody) to the orchestra.</p>
</li>
<li>
<p><strong>Forward Propagation (Performance)</strong>: As the orchestra plays, you guide each musician (neuron) to follow their musical notes (inputs), which are influenced by their respective instruments’ characteristics (weights) and personal preferences (biases). Each musician performs their part independently, and the sound of the entire orchestra (the model’s predictions) emerges.</p>
</li>
<li>
<p><strong>Loss Evaluation (Quality Assessment)</strong>: During the performance, you listen carefully to how the orchestra sounds (the model’s predictions). You compare the actual performance to the desired outcome (loss evaluation), assessing how well the orchestra played the piece.</p>
</li>
<li>
<p><strong>Backpropagation (Fine-Tuning)</strong>: Based on the feedback from the performance (loss evaluation), you provide guidance to each musician (neuron) on how to adjust their playing (weights and biases) to produce a better sound (minimize the loss).</p>
</li>
<li>
<p><strong>Iterative Learning (Repetition)</strong>: The rehearsal-performance-feedback cycle repeats iteratively. The orchestra continues to play the piece, fine-tuning their playing after each performance (training epochs). With each iteration, the orchestra’s performance improves, getting closer to the desired musical outcome.</p>
</li>
<li>
<p><strong>Convergence (Optimal Performance)</strong>: As the rehearsals and performances continue, the orchestra’s sound becomes more refined, and the desired outcome is achieved (model convergence). At this point, your orchestra (the neural network) has learned to play the musical piece accurately.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-a-neural-network">Mathematical Representation of a Neural Network</h3>
<p>Let’s represent the key components of a neural network in mathematical expressions:</p>
<ol>
<li><strong>Forward Propagation</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">forward_propagation</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for forward propagation)</span>
    <span class="token keyword">return</span> current_layer_outputs  <span class="token comment"># The final output of the neural network</span>
</code></pre>
<ol start="2">
<li><strong>Loss Calculation</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">calculate_loss</span><span class="token punctuation">(</span>predictions<span class="token punctuation">,</span> target_values<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for loss calculation)</span>
    <span class="token keyword">return</span> loss
</code></pre>
<ol start="3">
<li><strong>Backward Propagation</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">backward_propagation</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for backward propagation)</span>
    <span class="token keyword">return</span> output_gradients
</code></pre>
<ol start="4">
<li><strong>Parameter Updates</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">update_parameters</span><span class="token punctuation">(</span>weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> gradients<span class="token punctuation">,</span> learning_rate<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># ... (code for parameter updates)</span>
</code></pre>
<ol start="5">
<li><strong>Complete Training Loop</strong>:</li>
</ol>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">train_neural_network</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">,</span> learning_rate<span class="token punctuation">,</span> epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token keyword">for</span> epoch <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span>epochs<span class="token punctuation">)</span><span class="token punctuation">:</span>
        predictions <span class="token operator">=</span> forward_propagation<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>
        loss <span class="token operator">=</span> calculate_loss<span class="token punctuation">(</span>predictions<span class="token punctuation">,</span> target_values<span class="token punctuation">)</span>
        gradients <span class="token operator">=</span> backward_propagation<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> target_values<span class="token punctuation">,</span> weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> activation_derivative<span class="token punctuation">)</span>
        update_parameters<span class="token punctuation">(</span>weights_matrices<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> gradients<span class="token punctuation">,</span> learning_rate<span class="token punctuation">)</span>
    <span class="token keyword">return</span> weights_matrices<span class="token punctuation">,</span> biases
</code></pre>
<p>A neural network, like an orchestra, performs a task by combining the efforts of individual members (neurons) to produce a unified output. Through rehearsals (training) and iterative feedback (backpropagation), the neural network fine-tunes its parameters to make more accurate predictions. Just as a conductor guides an orchestra to a perfect performance, a well-trained neural network can achieve outstanding results by learning from data and adjusting its internal parameters accordingly.</p>
<h2 id="convolutional-neural-network-cnn">Convolutional Neural Network (CNN)</h2>
<h3 id="introduction-9">Introduction</h3>
<p>A Convolutional Neural Network (CNN) is a specialized type of neural network designed for image recognition, computer vision, and other tasks involving grid-like data. CNNs are inspired by the visual processing in the human brain and are highly effective in capturing spatial patterns and hierarchies of features from images. They consist of convolutional layers, pooling layers, and fully connected layers that work together to extract and process information from images.</p>
<h3 id="the-analogy-the-art-detective">The Analogy: The Art Detective</h3>
<p>To understand how a Convolutional Neural Network works, let’s use an analogy involving an art detective.</p>
<p>Imagine you are an art detective investigating a mysterious painting (the input image) with intricate details. To understand the painting better and uncover hidden patterns (features), you use different tools and techniques, such as magnifying glasses, pattern recognition, and combining smaller pieces to reconstruct the larger picture. Similarly, a CNN employs convolutional filters, pooling operations, and fully connected layers to extract features and make sense of the input image.</p>
<h3 id="how-a-cnn-works-in-detail">How a CNN Works in Detail</h3>
<ol>
<li>
<p><strong>Input Image</strong>: You start with the mysterious painting (the input image) that you want to analyze. The image is represented as a grid of pixel values, where each pixel represents a color or intensity.</p>
</li>
<li>
<p><strong>Convolutional Filters (Magnifying Glasses)</strong>: Just as you use magnifying glasses to zoom in on specific parts of the painting, a CNN uses convolutional filters (small windows with learnable weights) to scan and analyze different regions of the image. These filters act as magnifying glasses, focusing on local features and patterns.</p>
</li>
<li>
<p><strong>Convolution Operation (Pattern Recognition)</strong>: As the magnifying glass moves across the image, it detects patterns and features in the local regions. The convolution operation involves taking the dot product between the filter and the corresponding region of the image, producing feature maps that highlight specific patterns.</p>
</li>
<li>
<p><strong>Activation Function (Spotlight)</strong>: The feature maps go through an activation function, which introduces non-linearity and decides which features to highlight. The activation function acts like a spotlight, emphasizing important patterns while dimming irrelevant ones.</p>
</li>
<li>
<p><strong>Pooling Layers (Piece Reconstruction)</strong>: After detecting local features, you use pooling layers to downsample the feature maps. Pooling combines smaller pieces into more abstract representations, summarizing the detected patterns. This process reduces the network’s parameters and makes it more efficient.</p>
</li>
<li>
<p><strong>Fully Connected Layers (Art Analysis)</strong>: Finally, the abstract representations from the pooling layers are fed into fully connected layers, which act as the art analysis stage. These layers process the abstract features and make high-level predictions about the content of the painting (e.g., whether it depicts a landscape, portrait, etc.).</p>
</li>
<li>
<p><strong>Output (Art Conclusions)</strong>: The output layer produces the final predictions based on the analysis. It tells you the art’s genre, artist, or any other relevant information.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-convolutional-neural-network">Mathematical Representation of Convolutional Neural Network</h3>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Convolution Operation</span>
<span class="token keyword">def</span> <span class="token function">convolution_operation</span><span class="token punctuation">(</span>input_image<span class="token punctuation">,</span> convolutional_filter<span class="token punctuation">)</span><span class="token punctuation">:</span>
    feature_map <span class="token operator">=</span> convolution<span class="token punctuation">(</span>input_image<span class="token punctuation">,</span> convolutional_filter<span class="token punctuation">)</span>
    <span class="token keyword">return</span> feature_map

<span class="token comment"># Activation Function</span>
<span class="token keyword">def</span> <span class="token function">activation_function</span><span class="token punctuation">(</span>feature_map<span class="token punctuation">)</span><span class="token punctuation">:</span>
    activated_map <span class="token operator">=</span> relu<span class="token punctuation">(</span>feature_map<span class="token punctuation">)</span>
    <span class="token keyword">return</span> activated_map

<span class="token comment"># Pooling Layer</span>
<span class="token keyword">def</span> <span class="token function">pooling_layer</span><span class="token punctuation">(</span>input_map<span class="token punctuation">)</span><span class="token punctuation">:</span>
    pooled_map <span class="token operator">=</span> max_pooling<span class="token punctuation">(</span>input_map<span class="token punctuation">)</span>
    <span class="token keyword">return</span> pooled_map

<span class="token comment"># Fully Connected Layers (using dense layers)</span>
<span class="token keyword">def</span> <span class="token function">fully_connected_layers</span><span class="token punctuation">(</span>input_features<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">)</span><span class="token punctuation">:</span>
    output <span class="token operator">=</span> activation_function<span class="token punctuation">(</span>dot<span class="token punctuation">(</span>input_features<span class="token punctuation">,</span> weights<span class="token punctuation">)</span> <span class="token operator">+</span> biases<span class="token punctuation">)</span>
    <span class="token keyword">return</span> output
</code></pre>
<p>A Convolutional Neural Network acts like an art detective, analyzes input images using convolutional filters (magnifying glasses) to detect patterns and features. Pooling layers help reconstruct abstract representations, and fully connected layers perform the final analysis to make predictions about the image content. Just as an art detective combines tools to understand a painting’s essence, a CNN combines specialized layers to efficiently process images and learn complex visual representations.</p>
<h2 id="recurrent-neural-network-rnn">Recurrent Neural Network (RNN)</h2>
<h3 id="introduction-10">Introduction</h3>
<p>A Recurrent Neural Network (RNN) is a type of neural network designed to work with sequential data, such as time series or natural language. Unlike traditional feedforward neural networks, RNNs have connections that form cycles, allowing them to maintain hidden states and process sequences step by step. This enables RNNs to capture temporal dependencies and context in the data, making them powerful for tasks involving sequences.</p>
<h3 id="the-analogy-the-time-traveler">The Analogy: The Time Traveler</h3>
<p>To understand how a Recurrent Neural Network works, let’s use an analogy involving a time traveler exploring the past.</p>
<p>Imagine you are a time traveler equipped with a notebook (the RNN hidden state). As you journey through time (sequence of events), you observe and record important details and experiences in the notebook. As you continue traveling, you carry the knowledge from your past observations and use it to make sense of the events you encounter. This ability to retain and utilize past information allows you to better understand the context and dependencies between events.</p>
<h3 id="how-an-rnn-works-in-detail">How an RNN Works in Detail</h3>
<ol>
<li>
<p><strong>Input Sequence</strong>: The time traveler starts by observing a sequence of events (input sequence). Each event in the sequence has a specific context and influence on subsequent events.</p>
</li>
<li>
<p><strong>Hidden State (Notebook)</strong>: As the time traveler moves through the sequence, they carry a notebook (the hidden state) that retains information from previous events. The hidden state acts as the memory of the RNN, allowing it to capture past information and context.</p>
</li>
<li>
<p><strong>Processing Sequence (Time Traveling)</strong>: At each step in the sequence, the time traveler observes an event, updates their notebook (hidden state) with new information, and moves to the next event. They combine the knowledge from the past (hidden state) with the current event to make sense of the sequence.</p>
</li>
<li>
<p><strong>Information Fusion (Contextual Understanding)</strong>: Throughout the journey, the time traveler continually fuses new information with the context stored in their notebook (hidden state). This allows them to understand the temporal dependencies and patterns in the sequence.</p>
</li>
<li>
<p><strong>Output (Time Traveler’s Conclusion)</strong>: As the journey through the sequence concludes, the time traveler possesses a comprehensive understanding of the entire sequence. They have utilized their notebook (hidden state) to capture dependencies and context, making informed conclusions about the sequence.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-recurrent-neural-network">Mathematical Representation of Recurrent Neural Network</h3>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># RNN Cell (Single Time Step)</span>
<span class="token keyword">def</span> <span class="token function">rnn_cell</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># Combining input data with previous hidden state</span>
    combined_input <span class="token operator">=</span> dot<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'input'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> dot<span class="token punctuation">(</span>previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'hidden'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> biases
    new_hidden_state <span class="token operator">=</span> activation_function<span class="token punctuation">(</span>combined_input<span class="token punctuation">)</span>
    <span class="token keyword">return</span> new_hidden_state

<span class="token comment"># Complete RNN for a Sequence</span>
<span class="token keyword">def</span> <span class="token function">rnn_sequence</span><span class="token punctuation">(</span>input_sequence<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span><span class="token punctuation">:</span>
    hidden_states <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    previous_hidden_state <span class="token operator">=</span> initial_state  <span class="token comment"># Initial hidden state (notebook)</span>
    <span class="token keyword">for</span> time_step_data <span class="token keyword">in</span> input_sequence<span class="token punctuation">:</span>
        new_hidden_state <span class="token operator">=</span> rnn_cell<span class="token punctuation">(</span>time_step_data<span class="token punctuation">,</span> previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">)</span>
        hidden_states<span class="token punctuation">.</span>append<span class="token punctuation">(</span>new_hidden_state<span class="token punctuation">)</span>
        previous_hidden_state <span class="token operator">=</span> new_hidden_state  <span class="token comment"># Update the hidden state for the next time step</span>
    <span class="token keyword">return</span> hidden_states
</code></pre>
<p>The RNN’s ability to remember and utilize past information enables it to make informed conclusions about the sequence of events. Just as a time traveler moves through time, observing and recording experiences, an RNN moves through a sequence, processing data and maintaining hidden states to comprehend the temporal patterns in the data.</p>
<h2 id="long-short-term-memory-lstm">Long Short-Term Memory (LSTM)</h2>
<h3 id="introduction-11">Introduction</h3>
<p>Long Short-Term Memory (LSTM) is a specialized type of recurrent neural network (RNN) designed to address the vanishing gradient problem and capture long-range dependencies in sequential data. LSTMs have an additional architecture that allows them to remember and forget information selectively over time, making them powerful for tasks involving long sequences of data.</p>
<h3 id="the-analogy-the-organized-librarian">The Analogy: The Organized Librarian</h3>
<p>To understand how an LSTM works, let’s use an analogy involving an organized librarian managing a vast collection of books.</p>
<p>Imagine you are a librarian responsible for organizing a vast library (the sequential data). Each book represents a piece of information, and the shelves (memory cells) hold these books in different sections. Your job is to maintain an efficient system that allows you to quickly find the relevant books whenever needed (processing long sequences).</p>
<h3 id="how-an-lstm-works-in-detail">How an LSTM Works in Detail</h3>
<ol>
<li>
<p><strong>Book Checkout and Return (Input and Output)</strong>: People come to the library and borrow books (input data). After using the books, they return them to the library (output data). Similarly, an LSTM processes sequential data, accepting inputs at each time step and producing outputs accordingly.</p>
</li>
<li>
<p><strong>Book Organization (Memory Cell)</strong>: The librarian uses memory cells (memory blocks) to keep track of borrowed books. These cells have gates (controls) that allow the librarian to decide which books to keep, return, or add to the shelves.</p>
</li>
<li>
<p><strong>Selective Memory (Forget Gate)</strong>: When returning books, the librarian checks each memory cell and decides which books are no longer needed. The librarian forgets books that haven’t been borrowed recently. In the same way, an LSTM has a “forget gate” that selectively decides what information to forget from the memory cell.</p>
</li>
<li>
<p><strong>New Additions (Input Gate)</strong>: When new books are borrowed, the librarian decides which are worth keeping and adds them to the appropriate shelves. The librarian selectively accepts new information based on relevance. Similarly, an LSTM has an “input gate” that determines which new information to store in the memory cell.</p>
</li>
<li>
<p><strong>Recalling Information (Output Gate)</strong>: When someone requests specific books, the librarian fetches the relevant ones from the shelves and provides them. The librarian recalls information from the memory cell when needed. Similarly, an LSTM uses an “output gate” to decide which information to reveal as its output.</p>
</li>
<li>
<p><strong>Sequencing Information (Time Steps)</strong>: The librarian processes books in a specific order, based on borrowers’ requests and returns. Similarly, an LSTM processes sequential data one time step at a time, maintaining memory cell states and selectively processing new information.</p>
</li>
</ol>
<h3 id="mathematical-representation-of-long-short-term-memory-lstm">Mathematical Representation of Long Short-Term Memory (LSTM)</h3>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># LSTM Cell (Single Time Step)</span>
<span class="token keyword">def</span> <span class="token function">lstm_cell</span><span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> previous_hidden_state<span class="token punctuation">,</span> previous_cell_state<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> forget_gate_activation<span class="token punctuation">,</span> input_gate_activation<span class="token punctuation">,</span> output_gate_activation<span class="token punctuation">)</span><span class="token punctuation">:</span>
    <span class="token comment"># Calculate forget gate</span>
    forget_gate <span class="token operator">=</span> forget_gate_activation<span class="token punctuation">(</span>dot<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'input_forget'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> dot<span class="token punctuation">(</span>previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'hidden_forget'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> biases<span class="token punctuation">[</span><span class="token string">'forget'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    <span class="token comment"># Calculate input gate</span>
    input_gate <span class="token operator">=</span> input_gate_activation<span class="token punctuation">(</span>dot<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'input_input'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> dot<span class="token punctuation">(</span>previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'hidden_input'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> biases<span class="token punctuation">[</span><span class="token string">'input'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    <span class="token comment"># Calculate candidate cell state (new information)</span>
    candidate_cell_state <span class="token operator">=</span> activation_function<span class="token punctuation">(</span>dot<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'input_candidate'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> dot<span class="token punctuation">(</span>previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'hidden_candidate'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> biases<span class="token punctuation">[</span><span class="token string">'candidate'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    <span class="token comment"># Update cell state</span>
    cell_state <span class="token operator">=</span> forget_gate <span class="token operator">*</span> previous_cell_state <span class="token operator">+</span> input_gate <span class="token operator">*</span> candidate_cell_state
    <span class="token comment"># Calculate output gate</span>
    output_gate <span class="token operator">=</span> output_gate_activation<span class="token punctuation">(</span>dot<span class="token punctuation">(</span>input_data<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'input_output'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> dot<span class="token punctuation">(</span>previous_hidden_state<span class="token punctuation">,</span> weights<span class="token punctuation">[</span><span class="token string">'hidden_output'</span><span class="token punctuation">]</span><span class="token punctuation">)</span> <span class="token operator">+</span> biases<span class="token punctuation">[</span><span class="token string">'output'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
    <span class="token comment"># Update hidden state (output)</span>
    hidden_state <span class="token operator">=</span> output_gate <span class="token operator">*</span> activation_function<span class="token punctuation">(</span>cell_state<span class="token punctuation">)</span>

    <span class="token keyword">return</span> hidden_state<span class="token punctuation">,</span> cell_state

<span class="token comment"># Complete LSTM for a Sequence</span>
<span class="token keyword">def</span> <span class="token function">lstm_sequence</span><span class="token punctuation">(</span>input_sequence<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> forget_gate_activation<span class="token punctuation">,</span> input_gate_activation<span class="token punctuation">,</span> output_gate_activation<span class="token punctuation">)</span><span class="token punctuation">:</span>
    hidden_states <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    previous_hidden_state <span class="token operator">=</span> initial_state  <span class="token comment"># Initial hidden state</span>
    previous_cell_state <span class="token operator">=</span> initial_state    <span class="token comment"># Initial cell state</span>
    <span class="token keyword">for</span> time_step_data <span class="token keyword">in</span> input_sequence<span class="token punctuation">:</span>
        new_hidden_state<span class="token punctuation">,</span> new_cell_state <span class="token operator">=</span> lstm_cell<span class="token punctuation">(</span>time_step_data<span class="token punctuation">,</span> previous_hidden_state<span class="token punctuation">,</span> previous_cell_state<span class="token punctuation">,</span> weights<span class="token punctuation">,</span> biases<span class="token punctuation">,</span> activation_function<span class="token punctuation">,</span> forget_gate_activation<span class="token punctuation">,</span> input_gate_activation<span class="token punctuation">,</span> output_gate_activation<span class="token punctuation">)</span>
        hidden_states<span class="token punctuation">.</span>append<span class="token punctuation">(</span>new_hidden_state<span class="token punctuation">)</span>
        previous_hidden_state <span class="token operator">=</span> new_hidden_state      <span class="token comment"># Update hidden state for the next time step</span>
        previous_cell_state <span class="token operator">=</span> new_cell_state          <span class="token comment"># Update cell state for the next time step</span>
    <span class="token keyword">return</span> hidden_states
</code></pre>
<p>Basically, an LSTM, like an organized librarian, selectively processes sequential data using memory cells with gates to control information flow. The LSTM’s ability to remember and selectively update information allows it to capture long-range dependencies and context, making it a powerful tool for tasks involving sequential data. Just as an organized librarian efficiently manages books in a vast library, an LSTM effectively processes and retains relevant information from long sequences of data.</p>
<h2 id="how-a-cnn-works---a-mathematical-example">How a CNN Works - A Mathematical Example</h2>
<p>Let’s use a simple 3x3 grayscale image and explain how a Convolutional Neural Network (CNN) works with it step by step. For simplicity, we’ll use a single convolutional filter and a ReLU activation function.</p>
<h3 id="sample-image">Sample Image:</h3>
<pre><code>Image (3x3 grayscale):
  2  1  0
  1  0  2
  0  1  1
</code></pre>
<h3 id="convolutional-filter-kernel">Convolutional Filter (Kernel):</h3>
<pre><code>Filter (3x3):
  1  0  1
  0  1  0
  1  0  1
</code></pre>
<h3 id="convolution-operation">Convolution Operation:</h3>
<p>The convolution operation involves sliding the filter over the image and calculating the dot product at each position to obtain the feature map.</p>
<p>Let’s calculate the output at each position:</p>
<pre><code>(1*2) + (0*1) + (1*0) + (0*1) + (1*0) + (0*2) + (1*0) + (0*1) + (1*1) = 2 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 1 = 3

(0*2) + (1*1) + (0*0) + (1*1) + (0*0) + (1*2) + (0*0) + (1*1) + (0*1) = 0 + 1 + 0 + 1 + 0 + 2 + 0 + 1 + 0 = 5

(1*1) + (0*0) + (1*2) + (0*1) + (1*0) + (0*2) + (1*1) + (0*1) + (1*1) = 1 + 0 + 2 + 0 + 0 + 0 + 1 + 0 + 1 = 5

Feature Map (Output):
  3  5  5
  2  0  2
  5  5  3
</code></pre>
<h3 id="relu-activation-function">ReLU Activation Function:</h3>
<p>ReLU (Rectified Linear Activation Function) applies the element-wise activation on the feature map, setting all negative values to zero.</p>
<p>ReLU Output:</p>
<pre><code>ReLU(Feature Map):
  3  5  5
  2  0  2
  5  5  3
</code></pre>
<h3 id="explanation">Explanation:</h3>
<p>In this example, we performed the convolution operation between the 3x3 image and the 3x3 convolutional filter. The resulting feature map is obtained by sliding the filter over the image and calculating the dot product at each position. The ReLU activation function is then applied to the feature map, introducing non-linearity and transforming the negative values to zero.</p>
<p>This process is the essence of how a Convolutional Neural Network (CNN) works. CNNs utilize multiple convolutional filters with different weights to detect various patterns and features in images. These filters are learned during the training process to make accurate predictions and classify objects within images. CNNs are widely used in computer vision tasks, such as object detection, image recognition, and more, due to their ability to automatically learn hierarchical features from raw image data.</p>
<h2 id="how-an-rnn-works---a-mathematical-example">How an RNN Works - A Mathematical Example</h2>
<p>Sure! Let’s use a simple sequence of numbers and explain how a Recurrent Neural Network (RNN) works with it step by step. For simplicity, we’ll use a single RNN cell with a hidden state size of 2 and a tanh activation function.</p>
<h3 id="sample-sequence">Sample Sequence:</h3>
<pre><code>Sequence: [1, 2, 3, 4, 5]
</code></pre>
<h3 id="rnn-cell">RNN Cell:</h3>
<p>An RNN cell processes each element of the sequence one by one, maintaining a hidden state to capture dependencies between elements.</p>
<p>Let’s calculate the hidden state at each time step:</p>
<pre><code>Input: 1
Previous Hidden State: [0, 0]
Weights:
  - Input Weights: [0.1, 0.2]
  - Hidden State Weights: [0.3, 0.4]
Biases: [0.5, 0.6]

Combined Input: (1 * 0.1) + (0 * 0.3) + (0.1 * 0.2) + (0 * 0.4) + 0.5 = 0.1 + 0 + 0.02 + 0 + 0.5 = 0.62
New Hidden State: tanh(0.62) = 0.5433

Input: 2
Previous Hidden State: [0.5433, 0]
Weights:
  - Input Weights: [0.1, 0.2]
  - Hidden State Weights: [0.3, 0.4]
Biases: [0.5, 0.6]

Combined Input: (2 * 0.1) + (0.5433 * 0.3) + (0 * 0.2) + (0 * 0.4) + 0.6 = 0.2 + 0.16299 + 0 + 0 + 0.6 = 0.9633
New Hidden State: tanh(0.9633) = 0.7427

Input: 3
Previous Hidden State: [0.7427, 0]
Weights:
  - Input Weights: [0.1, 0.2]
  - Hidden State Weights: [0.3, 0.4]
Biases: [0.5, 0.6]

Combined Input: (3 * 0.1) + (0.7427 * 0.3) + (0 * 0.2) + (0 * 0.4) + 0.6 = 0.3 + 0.22281 + 0 + 0 + 0.6 = 1.12281
New Hidden State: tanh(1.12281) = 0.8197

Input: 4
Previous Hidden State: [0.8197, 0]
Weights:
  - Input Weights: [0.1, 0.2]
  - Hidden State Weights: [0.3, 0.4]
Biases: [0.5, 0.6]

Combined Input: (4 * 0.1) + (0.8197 * 0.3) + (0 * 0.2) + (0 * 0.4) + 0.6 = 0.4 + 0.24591 + 0 + 0 + 0.6 = 1.24591
New Hidden State: tanh(1.24591) = 0.8482

Input: 5
Previous Hidden State: [0.8482, 0]
Weights:
  - Input Weights: [0.1, 0.2]
  - Hidden State Weights: [0.3, 0.4]
Biases: [0.5, 0.6]

Combined Input: (5 * 0.1) + (0.8482 * 0.3) + (0 * 0.2) + (0 * 0.4) + 0.6 = 0.5 + 0.25446 + 0 + 0 + 0.6 = 1.35446
New Hidden State: tanh(1.35446) = 0.8809
</code></pre>
<h3 id="explanation-1">Explanation:</h3>
<p>In this example, we used a simple sequence [1, 2, 3, 4, 5] and processed it using an RNN cell with a hidden state size of 2. At each time step, the RNN cell takes the input element and the previous hidden state as input, combines them with weights and biases, and applies the tanh activation function to calculate the new hidden state.</p>
<p>This process is the essence of how a Recurrent Neural Network (RNN) works. RNNs process sequential data one element at a time while maintaining a hidden state that captures dependencies between elements. The hidden state serves as the memory of the RNN, enabling it to learn temporal patterns and context from the sequence. RNNs are commonly used in tasks involving time series data, natural language processing, and other sequential data where capturing temporal dependencies is crucial.</p>
<h2 id="how-an-lstm-works---a-mathematical-example">How an LSTM Works - A Mathematical Example</h2>
<p>We are going to use a simple sequence of numbers as an image and explain how a Long Short-Term Memory (LSTM) works with it step by step. For simplicity, we’ll use a single LSTM cell with a hidden state size of 2 and a tanh activation function.</p>
<h3 id="sample-sequence-1">Sample Sequence:</h3>
<pre><code>Sequence: [1, 2, 3, 4, 5]
</code></pre>
<h3 id="lstm-cell">LSTM Cell:</h3>
<p>An LSTM cell processes each element of the sequence one by one, maintaining both a hidden state and a cell state to capture long-term dependencies between elements.</p>
<p>Let’s calculate the hidden state and cell state at each time step:</p>
<pre><code>Input: 1
Previous Hidden State: [0, 0]
Previous Cell State: [0, 0]

Input Weights:
  - Forget Gate: [0.1, 0.2]
  - Input Gate: [0.3, 0.4]
  - Candidate Cell State: [0.5, 0.6]
  - Output Gate: [0.7, 0.8]

Hidden State Weights:
  - Forget Gate: [0.9, 1.0]
  - Input Gate: [1.1, 1.2]
  - Candidate Cell State: [1.3, 1.4]
  - Output Gate: [1.5, 1.6]

Biases:
  - Forget Gate: 0.17
  - Input Gate: 0.18
  - Candidate Cell State: 0.19
  - Output Gate: 0.20

Combined Input:
  Forget Gate: (1 * 0.1) + (0 * 0.9) + (0 * 0.2) + (0 * 1.0) + 0.17 = 0.1 + 0 + 0 + 0 + 0.17 = 0.27
  Input Gate: (1 * 0.3) + (0 * 1.1) + (0 * 0.4) + (0 * 1.2) + 0.18 = 0.3 + 0 + 0 + 0 + 0.18 = 0.48
  Candidate Cell State: (1 * 0.5) + (0 * 1.3) + (0 * 0.6) + (0 * 1.4) + 0.19 = 0.5 + 0 + 0 + 0 + 0.19 = 0.69
  Output Gate: (1 * 0.7) + (0 * 1.5) + (0 * 0.8) + (0 * 1.6) + 0.20 = 0.7 + 0 + 0 + 0 + 0.20 = 0.90

New Cell State:
  Forget Gate: 0.27 * 0 + 0 * 0 = 0
  Input Gate: 0.48 * 0.69 + 0 * 0 = 0.3312
  Candidate Cell State: 0.69 * 1 + 0 * 0 = 0.69
  Output Gate: 0.90 * 0.69 + 0 * 0 = 0.621

New Hidden State:
  tanh(0.621) = 0.5456

Input: 2
Previous Hidden State: [0.5456, 0]
Previous Cell State: [0, 0.3312]

Combined Input:
  Forget Gate: (2 * 0.1) + (0.5456 * 0.9) + (0 * 0.2) + (0.3312 * 1.0) + 0.17 = 0.2 + 0.49104 + 0 + 0.3312 + 0.17 = 1.19224
  Input Gate: (2 * 0.3) + (0.5456 * 1.1) + (0 * 0.4) + (0.3312 * 1.2) + 0.18 = 0.6 + 0.60016 + 0 + 0.39744 + 0.18 = 1.7776
  Candidate Cell State: (2 * 0.5) + (0.5456 * 1.3) + (0 * 0.6) + (0.3312 * 1.4) + 0.19 = 1 + 0.70928 + 0 + 0.46368 + 0.19 = 2.36296
  Output Gate: (2 * 0.7) + (0.5456 * 1.5) + (0 * 0.8) + (0.3312 * 1.6) + 0.20 = 1.4 + 0.8184 + 0 + 0.52992 + 0.20 = 2.94832

New Cell State:
  Forget Gate: 1 * 0 + 0.3312 * 0.27 = 0
  Input Gate: 0.3312 * 2.36296 + 0 * 0.48 = 0.782128
  Candidate Cell State: 0.782128 + 0.3312 * 0.69 = 1.004688
  Output Gate: 1.004688 * 0.621 + 0 * 0.90 = 0.62353

New Hidden State:
  tanh(0.62353) = 0.5596

Repeat the above calculations for the rest of the sequence...

</code></pre>
<h3 id="explanation-2">Explanation:</h3>
<p>In this example, we used a simple sequence [1, 2, 3, 4, 5] and processed it using an LSTM cell with a hidden state size of 2. At each time step, the LSTM cell takes the input element, the previous hidden state, and the previous cell state as input. It then combines them with different weights and biases to calculate the new hidden state and the new cell state.</p>
<p>This process is the essence of how a Long Short-Term Memory (LSTM) works. LSTMs process sequential data one element at a time while maintaining both a hidden state and a cell state that capture long-term dependencies between elements. The LSTM’s ability to selectively update the cell state using forget and input gates enables it to learn and remember long-range patterns and context from the sequence. LSTMs are widely used in tasks involving time series data, natural language processing, and other sequential data where capturing long-term dependencies is essential.</p>
<h2 id="explanation-of-the-rnn-for-electricity-theft-detection-code">Explanation of the RNN for Electricity Theft Detection Code</h2>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
<span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> train_test_split
<span class="token keyword">import</span> tensorflow <span class="token keyword">as</span> tf
<span class="token keyword">from</span> tensorflow<span class="token punctuation">.</span>keras <span class="token keyword">import</span> layers<span class="token punctuation">,</span> models
</code></pre>
<ul>
<li>The code starts by importing the necessary libraries: pandas (for data manipulation), numpy (for numerical operations), train_test_split from scikit-learn (for splitting data into training and testing sets), and the required modules from TensorFlow (for building the neural network).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">if</span> data<span class="token punctuation">.</span>isnull<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">.</span>values<span class="token punctuation">.</span><span class="token builtin">any</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    data <span class="token operator">=</span> data<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code checks if there are any null values in the ‘data’ DataFrame using the <code>data.isnull().values.any()</code> function. If there are any null values, the code fills them with 0 using the <code>data.fillna(0)</code> method.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">labels <span class="token operator">=</span> data<span class="token punctuation">[</span><span class="token string">'FLAG'</span><span class="token punctuation">]</span>
input_features <span class="token operator">=</span> data<span class="token punctuation">.</span>drop<span class="token punctuation">(</span>columns<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'CUSTOMER'</span><span class="token punctuation">,</span> <span class="token string">'FLAG'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code separates the labels from the input features. The ‘labels’ variable contains the ‘FLAG’ column, which indicates whether the customer is fraudulent (1) or normal (0). The ‘input_features’ variable contains the remaining columns from the ‘data’ DataFrame, excluding the ‘CUSTOMER’ and ‘FLAG’ columns.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>input_features<span class="token punctuation">,</span> labels<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.3</span><span class="token punctuation">,</span> stratify<span class="token operator">=</span>labels<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses the <code>train_test_split</code> function from scikit-learn to split the data into training and testing sets. The input features (X) and labels (y) are divided into X_train, X_test, y_train, and y_test. The test_size parameter is set to 0.3, indicating that 30% of the data will be used for testing, and the remaining 70% will be used for training. The stratify parameter ensures that the class distribution in the training and testing sets is similar to the original class distribution (fraudulent vs. normal customers). The random_state is set to 42 for reproducibility.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">X_train_rnn <span class="token operator">=</span> X_train<span class="token punctuation">.</span>values
X_test_rnn <span class="token operator">=</span> X_test<span class="token punctuation">.</span>values
</code></pre>
<ul>
<li>The code converts the training and testing input features (X_train and X_test) to numpy arrays using the <code>.values</code> attribute. TensorFlow requires data to be in numpy array format for training.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">time_steps <span class="token operator">=</span> <span class="token number">12</span>
</code></pre>
<ul>
<li>The code sets the number of time steps for the RNN to 12. The ‘time_steps’ parameter determines how many consecutive data points the RNN will process together as a sequence.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">create_sequences</span><span class="token punctuation">(</span>data<span class="token punctuation">,</span> time_steps<span class="token punctuation">)</span><span class="token punctuation">:</span>
    sequences <span class="token operator">=</span> <span class="token punctuation">[</span><span class="token punctuation">]</span>
    <span class="token keyword">for</span> i <span class="token keyword">in</span> <span class="token builtin">range</span><span class="token punctuation">(</span><span class="token builtin">len</span><span class="token punctuation">(</span>data<span class="token punctuation">)</span> <span class="token operator">-</span> time_steps <span class="token operator">+</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
        sequences<span class="token punctuation">.</span>append<span class="token punctuation">(</span>data<span class="token punctuation">[</span>i<span class="token punctuation">:</span>i <span class="token operator">+</span> time_steps<span class="token punctuation">]</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>sequences<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code defines a function named <code>create_sequences</code> that takes the input data and ‘time_steps’ as input and returns sequences of data with the specified time_steps. This function creates overlapping sequences from the input data. For example, if time_steps is 12, it creates sequences of 12 consecutive data points from the input data.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">X_train_rnn <span class="token operator">=</span> create_sequences<span class="token punctuation">(</span>X_train_rnn<span class="token punctuation">,</span> time_steps<span class="token punctuation">)</span>
X_test_rnn <span class="token operator">=</span> create_sequences<span class="token punctuation">(</span>X_test_rnn<span class="token punctuation">,</span> time_steps<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code calls the <code>create_sequences</code> function on the training and testing input features (X_train_rnn and X_test_rnn) to create sequences suitable for RNN processing. The training and testing data are now represented as 3D arrays, with dimensions (number_of_sequences, time_steps, number_of_features).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">y_train <span class="token operator">=</span> y_train<span class="token punctuation">[</span>time_steps <span class="token operator">-</span> <span class="token number">1</span><span class="token punctuation">:</span><span class="token punctuation">]</span><span class="token punctuation">.</span>to_numpy<span class="token punctuation">(</span><span class="token punctuation">)</span>
y_test <span class="token operator">=</span> y_test<span class="token punctuation">[</span>time_steps <span class="token operator">-</span> <span class="token number">1</span><span class="token punctuation">:</span><span class="token punctuation">]</span><span class="token punctuation">.</span>to_numpy<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code adjusts the labels (y_train and y_test) to match the corresponding X data after creating sequences. The first (time_steps - 1) labels are removed to match the number of sequences.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">create_rnn_model</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    model <span class="token operator">=</span> models<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>SimpleRNN<span class="token punctuation">(</span><span class="token number">128</span><span class="token punctuation">,</span> input_shape<span class="token operator">=</span><span class="token punctuation">(</span>X_train_rnn<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span><span class="token punctuation">,</span> X_train_rnn<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">2</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'sigmoid'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>  <span class="token comment"># Output layer with sigmoid for binary classification (theft or normal)</span>
    <span class="token keyword">return</span> model
</code></pre>
<ul>
<li>The code defines a function named <code>create_rnn_model()</code> that returns the architecture of the RNN model. The model consists of a SimpleRNN layer with 128 units, followed by a Dense layer with 64 units and ReLU activation function, and finally, an output layer with a single unit and a sigmoid activation function for binary classification (fraudulent or normal).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">rnn_model <span class="token operator">=</span> create_rnn_model<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code calls the <code>create_rnn_model()</code> function to create the RNN model and assigns it to the variable <code>rnn_model</code>.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">rnn_model<span class="token punctuation">.</span><span class="token builtin">compile</span><span class="token punctuation">(</span>optimizer<span class="token operator">=</span><span class="token string">'adam'</span><span class="token punctuation">,</span>
                  loss<span class="token operator">=</span><span class="token string">'binary_crossentropy'</span><span class="token punctuation">,</span>
                  metrics<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code compiles the RNN model using the ‘adam’ optimizer, ‘binary_crossentropy’ loss function (suitable for binary classification), and ‘accuracy’ metric for model training.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">rnn_model<span class="token punctuation">.</span>summary<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code prints a summary of the RNN model, including the layer configurations and the number of parameters.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">test_loss_before<span class="token punctuation">,</span> test_accuracy_before <span class="token operator">=</span> rnn_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_rnn<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy before training:"</span><span class="token punctuation">,</span> test_accuracy_before<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the model on the test data before training using the <code>evaluate</code> method. The accuracy and loss are stored in <code>test_accuracy_before</code> and <code>test_loss_before</code> variables, respectively, and then printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">history <span class="token operator">=</span> rnn_model<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train_rnn<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> epochs<span class="token operator">=</span><span class="token number">15</span><span class="token punctuation">,</span> batch_size<span class="token operator">=</span>

<span class="token number">32</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code trains the RNN model using the training data (X_train_rnn and y_train) with 15 epochs and a batch size of 32. The training history is stored in the ‘history’ variable.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">test_loss<span class="token punctuation">,</span> test_accuracy <span class="token operator">=</span> rnn_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_rnn<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy after training:"</span><span class="token punctuation">,</span> test_accuracy<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the model on the test data after training using the <code>evaluate</code> method. The updated accuracy and loss values are stored in <code>test_accuracy</code> and <code>test_loss</code> variables, respectively, and then printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'loss'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses matplotlib to plot the accuracy and loss over epochs to visualize the model’s performance during training. Two separate plots are generated for accuracy and loss, showing the trend of these metrics over the training epochs.</li>
</ul>
<h2 id="explanation-of-the-1d-cnn-for-electricity-theft-detection-code">Explanation of the 1D CNN for Electricity Theft Detection Code</h2>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
<span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> train_test_split
<span class="token keyword">import</span> tensorflow <span class="token keyword">as</span> tf
<span class="token keyword">from</span> tensorflow<span class="token punctuation">.</span>keras <span class="token keyword">import</span> layers<span class="token punctuation">,</span> models
</code></pre>
<ul>
<li>The code starts by importing the necessary libraries: pandas (for data manipulation), numpy (for numerical operations), train_test_split from scikit-learn (for splitting data into training and testing sets), and the required modules from TensorFlow for building the Convolutional Neural Network (CNN).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">labels <span class="token operator">=</span> data<span class="token punctuation">[</span><span class="token string">'FLAG'</span><span class="token punctuation">]</span>
input_features <span class="token operator">=</span> data<span class="token punctuation">.</span>drop<span class="token punctuation">(</span>columns<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'CUSTOMER'</span><span class="token punctuation">,</span> <span class="token string">'FLAG'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code separates the labels from the input features. The ‘labels’ variable contains the ‘FLAG’ column, which indicates whether the customer is fraudulent (1) or normal (0). The ‘input_features’ variable contains the remaining columns from the ‘data’ DataFrame, excluding the ‘CUSTOMER’ and ‘FLAG’ columns.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">input_features <span class="token operator">=</span> input_features<span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span>pd<span class="token punctuation">.</span>to_numeric<span class="token punctuation">,</span> errors<span class="token operator">=</span><span class="token string">'coerce'</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code converts the input features to numeric data by applying <code>pd.to_numeric</code> with the ‘coerce’ option. This ensures that non-numeric values in the input features are converted to NaN (Not a Number).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">input_features <span class="token operator">=</span> input_features<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code fills any NaN values in the input features with 0 using the <code>fillna</code> method. This step is essential for ensuring that all data is in numeric format and ready for further processing.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>input_features<span class="token punctuation">,</span> labels<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.3</span><span class="token punctuation">,</span> stratify<span class="token operator">=</span>labels<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses the <code>train_test_split</code> function from scikit-learn to split the data into training and testing sets. The input features (X) and labels (y) are divided into X_train, X_test, y_train, and y_test. The test_size parameter is set to 0.3, indicating that 30% of the data will be used for testing, and the remaining 70% will be used for training. The stratify parameter ensures that the class distribution in the training and testing sets is similar to the original class distribution (fraudulent vs. normal customers). The random_state is set to 42 for reproducibility.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">num_features <span class="token operator">=</span> X_train<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span>
X_train_cnn <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_train<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
X_test_cnn <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code reshapes the training and testing input features (X_train and X_test) for CNN processing. For CNN, the input data needs to have a 3D shape (batch_size, sequence_length, channels). In this case, the batch_size is set to -1 (automatically determined based on the data size), sequence_length is the number of features in each input (num_features), and channels is set to 1 for grayscale images.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">create_cnn_model</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    model <span class="token operator">=</span> models<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">32</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> input_shape<span class="token operator">=</span><span class="token punctuation">(</span>num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">128</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">256</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">512</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Flatten<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'sigmoid'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> model
</code></pre>
<ul>
<li>The code defines a function named <code>create_cnn_model()</code> that returns the architecture of the CNN model. The model consists of several Conv1D layers with different numbers of filters and kernel sizes, followed by MaxPooling1D layers to downsample the data. The final layers consist of a Flatten layer to flatten the output and two Dense layers with ReLU and sigmoid activations for binary classification (fraudulent or normal).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">cnn_model <span class="token operator">=</span> create_cnn_model<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code calls the <code>create_cnn_model()</code> function to create the CNN model and assigns it to the variable <code>cnn_model</code>.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">cnn_model<span class="token punctuation">.</span><span class="token builtin">compile</span><span class="token punctuation">(</span>optimizer<span class="token operator">=</span><span class="token string">'adam'</span><span class="token punctuation">,</span> loss<span class="token operator">=</span><span class="token string">'binary_crossentropy'</span><span class="token punctuation">,</span> metrics<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code compiles the CNN model using the ‘adam’ optimizer, ‘binary_crossentropy’ loss function (suitable for binary classification), and ‘accuracy’ metric for model training.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">cnn_model<span class="token punctuation">.</span>summary<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code prints a summary of the CNN model, including the layer configurations and the number of parameters.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">test_loss_before<span class="token punctuation">,</span> test_accuracy_before <span class="token operator">=</span> cnn_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_cnn<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy before training:"</span><span class="token punctuation">,</span> test_accuracy_before<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the model on the test data before training using the <code>evaluate</code> method. The accuracy and loss are stored in <code>test_accuracy_before</code> and <code>test_loss_before</code> variables, respectively, and then printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">history <span class="token operator">=</span> cnn_model<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train_cnn<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> epochs<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> batch_size<span class="token operator">=</span><span class="token number">32</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code trains the CNN model using the training data (X_train_cnn and y_train) with 10 epochs and a batch size of 32. The training history is stored in the ‘history’ variable.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">test_loss<span class="token punctuation">,</span> test_accuracy <span class="token operator">=</span> cnn_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_cnn<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy:"</span><span class="token punctuation">,</span> test_accuracy<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the model on the test data after training using the <code>evaluate</code> method. The updated accuracy and loss values are stored in <code>test_accuracy</code> and <code>test_loss</code> variables, respectively, and then printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'loss'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses matplotlib to plot the accuracy and loss over epochs to visualize the model’s performance during training. Two separate plots are generated for accuracy and loss, showing the trend of these metrics over the training epochs.</li>
</ul>
<h2 id="explanation-of-the-lstm-for-electricity-theft-detection-code">Explanation of the LSTM for Electricity Theft Detection Code</h2>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
<span class="token keyword">import</span> tensorflow <span class="token keyword">as</span> tf
<span class="token keyword">from</span> tensorflow<span class="token punctuation">.</span>keras <span class="token keyword">import</span> layers<span class="token punctuation">,</span> models
</code></pre>
<ul>
<li>The code starts by importing the necessary libraries: pandas (for data manipulation), numpy (for numerical operations), and the required modules from TensorFlow for building the LSTM model.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">labels <span class="token operator">=</span> data<span class="token punctuation">[</span><span class="token string">'FLAG'</span><span class="token punctuation">]</span>
input_features <span class="token operator">=</span> data<span class="token punctuation">.</span>drop<span class="token punctuation">(</span>columns<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'CUSTOMER'</span><span class="token punctuation">,</span> <span class="token string">'FLAG'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code separates the labels from the input features. The ‘labels’ variable contains the ‘FLAG’ column, which indicates whether the customer is fraudulent (1) or normal (0). The ‘input_features’ variable contains the remaining columns from the ‘data’ DataFrame, excluding the ‘CUSTOMER’ and ‘FLAG’ columns.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">input_features <span class="token operator">=</span> input_features<span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span>pd<span class="token punctuation">.</span>to_numeric<span class="token punctuation">,</span> errors<span class="token operator">=</span><span class="token string">'coerce'</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code converts the input features to numeric data by applying <code>pd.to_numeric</code> with the ‘coerce’ option. This ensures that non-numeric values in the input features are converted to NaN (Not a Number).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">input_features <span class="token operator">=</span> input_features<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code fills any NaN values in the input features with 0 using the <code>fillna</code> method. This step is essential for ensuring that all data is in numeric format and ready for further processing.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>input_features<span class="token punctuation">,</span> labels<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.3</span><span class="token punctuation">,</span> stratify<span class="token operator">=</span>labels<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses the <code>train_test_split</code> function from scikit-learn to split the data into training and testing sets. The input features (X) and labels (y) are divided into X_train, X_test, y_train, and y_test. The test_size parameter is set to 0.3, indicating that 30% of the data will be used for testing, and the remaining 70% will be used for training. The stratify parameter ensures that the class distribution in the training and testing sets is similar to the original class distribution (fraudulent vs. normal customers). The random_state is set to 42 for reproducibility.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">time_steps <span class="token operator">=</span> <span class="token number">1</span>  <span class="token comment"># Each month considered as a single time step</span>
num_features <span class="token operator">=</span> X_train<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span>
X_train_lstm <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_train<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> time_steps<span class="token punctuation">,</span> num_features<span class="token punctuation">)</span>
X_test_lstm <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> time_steps<span class="token punctuation">,</span> num_features<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code reshapes the training and testing input features (X_train and X_test) for LSTM processing. For LSTM, the input data needs to have a 3D shape (batch_size, time_steps, num_features). In this case, the batch_size is set to -1 (automatically determined based on the data size), time_steps is set to 1 (as each month is considered a single time step), and num_features is the number of input features.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">def</span> <span class="token function">create_lstm_model</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    model <span class="token operator">=</span> models<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>LSTM<span class="token punctuation">(</span><span class="token number">128</span><span class="token punctuation">,</span> input_shape<span class="token operator">=</span><span class="token punctuation">(</span>time_steps<span class="token punctuation">,</span> num_features<span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'sigmoid'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>  <span class="token comment"># Output layer with sigmoid for binary classification</span>
    <span class="token keyword">return</span> model
</code></pre>
<ul>
<li>The code defines a function named <code>create_lstm_model()</code> that returns the architecture of the LSTM model. The model consists of an LSTM layer with 128 units and an input shape of (time_steps, num_features). The LSTM layer is followed by a Dense layer with 64 units and ReLU activation function, and finally, an output layer with a single unit and a sigmoid activation function for binary classification (fraudulent or normal).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">lstm_model <span class="token operator">=</span> create_lstm_model<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code calls the <code>create_lstm_model()</code> function to create the LSTM model and assigns it to the variable <code>lstm_model</code>.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">lstm_model<span class="token punctuation">.</span><span class="token builtin">compile</span><span class="token punctuation">(</span>optimizer<span class="token operator">=</span><span class="token string">'adam'</span><span class="token punctuation">,</span> loss<span class="token operator">=</span><span class="token string">'binary_crossentropy'</span><span class="token punctuation">,</span> metrics<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code compiles the LSTM model using the ‘adam’ optimizer, ‘binary_crossentropy’ loss function (suitable for binary classification), and ‘accuracy’ metric for model training.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">lstm_model<span class="token punctuation">.</span>summary<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code prints a summary of the LSTM model, including the layer configurations and the number of parameters.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">test_loss_before<span class="token punctuation">,</span> test_accuracy_before <span class="token operator">=</span> lstm_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_lstm<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy before training:"</span><span class="token punctuation">,</span> test_accuracy_before<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the model on the test data before training using the <code>evaluate</code> method. The accuracy and loss are stored in <code>test_accuracy_before</code> and <code>test_loss_before</code> variables, respectively, and then printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">history <span class="token operator">=</span> lstm_model<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train_lstm<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> epochs<span class="token operator">=</span><span class="token number">10</span><span class="token punctuation">,</span> batch_size<span class="token operator">=</span><span class="token number">32</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code trains the LSTM model using the training data (X_train_lstm and y_train) with 10 epochs and a batch size of 32. The training history is stored in the ‘history’ variable.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python">test_loss<span class="token punctuation">,</span> test_accuracy <span class="token operator">=</span> lstm_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_lstm

<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy:"</span><span class="token punctuation">,</span> test_accuracy<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the model on the test data after training using the <code>evaluate</code> method. The updated accuracy and loss values are stored in <code>test_accuracy</code> and <code>test_loss</code> variables, respectively, and then printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'loss'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses matplotlib to plot the accuracy and loss over epochs to visualize the model’s performance during training. Two separate plots are generated for accuracy and loss, showing the trend of these metrics over the training epochs.</li>
</ul>
<h2 id="explanation-of-the-hybrid-cnn-lstm-for-electricity-theft-detection-code">Explanation of the Hybrid CNN-LSTM for Electricity Theft Detection Code</h2>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Importing Dependencies</span>
<span class="token keyword">import</span> pandas <span class="token keyword">as</span> pd
<span class="token keyword">import</span> numpy <span class="token keyword">as</span> np
<span class="token keyword">import</span> tensorflow <span class="token keyword">as</span> tf
<span class="token keyword">from</span> tensorflow<span class="token punctuation">.</span>keras <span class="token keyword">import</span> layers<span class="token punctuation">,</span> models
<span class="token keyword">from</span> sklearn<span class="token punctuation">.</span>model_selection <span class="token keyword">import</span> train_test_split
</code></pre>
<ul>
<li>The code starts by importing the necessary libraries: pandas (for data manipulation), numpy (for numerical operations), and the required modules from TensorFlow for building the CNN and LSTM models. It also imports the <code>train_test_split</code> function from scikit-learn to split the data into training and testing sets.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Import the data</span>
data <span class="token operator">=</span> pd<span class="token punctuation">.</span>read_csv<span class="token punctuation">(</span><span class="token string">'Combined Customers Data - Sheet1.csv'</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code reads the data from a CSV file named ‘Combined Customers Data - Sheet1.csv’ and stores it in a pandas DataFrame called <code>data</code>.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Separate the labels (fraudulent or normal) from the input features</span>
labels <span class="token operator">=</span> data<span class="token punctuation">[</span><span class="token string">'FLAG'</span><span class="token punctuation">]</span>
input_features <span class="token operator">=</span> data<span class="token punctuation">.</span>drop<span class="token punctuation">(</span>columns<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'CUSTOMER'</span><span class="token punctuation">,</span> <span class="token string">'FLAG'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code separates the labels from the input features. The ‘labels’ variable contains the ‘FLAG’ column, which indicates whether the customer is fraudulent (1) or normal (0). The ‘input_features’ variable contains the remaining columns from the ‘data’ DataFrame, excluding the ‘CUSTOMER’ and ‘FLAG’ columns.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Replace non-numeric or null values with NaN in input features</span>
input_features <span class="token operator">=</span> input_features<span class="token punctuation">.</span><span class="token builtin">apply</span><span class="token punctuation">(</span>pd<span class="token punctuation">.</span>to_numeric<span class="token punctuation">,</span> errors<span class="token operator">=</span><span class="token string">'coerce'</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code converts the input features to numeric data by applying <code>pd.to_numeric</code> with the ‘coerce’ option. This ensures that non-numeric values in the input features are converted to NaN (Not a Number).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Replace NaN values with zero</span>
input_features <span class="token operator">=</span> input_features<span class="token punctuation">.</span>fillna<span class="token punctuation">(</span><span class="token number">0</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code fills any NaN values in the input features with 0 using the <code>fillna</code> method. This step is essential for ensuring that all data is in numeric format and ready for further processing.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Stratified splitting of the data into training and testing sets</span>
X_train<span class="token punctuation">,</span> X_test<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> y_test <span class="token operator">=</span> train_test_split<span class="token punctuation">(</span>input_features<span class="token punctuation">,</span> labels<span class="token punctuation">,</span> test_size<span class="token operator">=</span><span class="token number">0.3</span><span class="token punctuation">,</span> stratify<span class="token operator">=</span>labels<span class="token punctuation">,</span> random_state<span class="token operator">=</span><span class="token number">42</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses the <code>train_test_split</code> function from scikit-learn to split the data into training and testing sets. The input features (X) and labels (y) are divided into X_train, X_test, y_train, and y_test. The test_size parameter is set to 0.3, indicating that 30% of the data will be used for testing, and the remaining 70% will be used for training. The stratify parameter ensures that the class distribution in the training and testing sets is similar to the original class distribution (fraudulent vs. normal customers). The random_state is set to 42 for reproducibility.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Reshape the input data for CNN (assuming each month represents an image)</span>
num_features <span class="token operator">=</span> X_train<span class="token punctuation">.</span>shape<span class="token punctuation">[</span><span class="token number">1</span><span class="token punctuation">]</span>
X_train_cnn <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_train<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
X_test_cnn <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code reshapes the training and testing input features (X_train and X_test) for CNN processing. For CNN, the input data needs to have a 3D shape (batch_size, sequence_length, channels). In this case, the batch_size is set to -1 (automatically determined based on the data size), sequence_length is the number of features in each input (num_features), and channels is set to 1 (assuming each month is represented as an image in grayscale format).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Define the CNN model</span>
<span class="token keyword">def</span> <span class="token function">create_cnn_model</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    model <span class="token operator">=</span> models<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">32</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> input_shape<span class="token operator">=</span><span class="token punctuation">(</span>num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">128</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">256</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Conv1D<span class="token punctuation">(</span><span class="token number">512</span><span class="token punctuation">,</span> <span class="token number">3</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">,</span> padding<span class="token operator">=</span><span class="token string">'same'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>MaxPooling1D<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Flatten<span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> model
</code></pre>
<ul>
<li>The code defines a function named <code>create_cnn_model()</code> that returns the architecture of the CNN model. The model consists of several Conv1D layers with different numbers of filters and kernel sizes, followed by MaxPooling1D layers to downsample the data. The final layers consist of a Flatten layer to flatten the output.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Create the CNN model</span>
cnn_model <span class="token operator">=</span> create_cnn_model<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code calls the <code>create_cnn_model()</code> function to create the CNN model and assigns it to the variable <code>cnn_model</code>.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Define the LSTM model</span>
<span class="token keyword">def</span> <span class="token function">create_lstm_model</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">:</span>
    model <span class="token operator">=</span> models<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>LSTM<span class="token punctuation">(</span><span class="token number">128</span><span class="token punctuation">,</span> input_shape<span class="token operator">=</span><span class="token punctuation">(</span>num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'relu'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'sigmoid'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
    <span class="token keyword">return</span> model
</code></pre>
<ul>
<li>The code defines a function named <code>create_lstm_model()</code> that returns the architecture of the LSTM model. The model consists of an LSTM layer with 128 units and an input shape of (num_features, 1). The LSTM layer is followed by a Dense layer with 64 units and ReLU activation function, and finally, an output layer with a single unit and a sigmoid activation function for binary classification (fraudulent or normal).</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Create the LSTM model</span>
lstm_model <span class="token operator">=</span> create_lstm_model<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code calls the `create_lstm</li>
</ul>
<p>_model()<code>function to create the LSTM model and assigns it to the variable</code>lstm_model`.</p>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Combine the CNN and LSTM models</span>
combined_model <span class="token operator">=</span> models<span class="token punctuation">.</span>Sequential<span class="token punctuation">(</span><span class="token punctuation">)</span>
combined_model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>TimeDistributed<span class="token punctuation">(</span>cnn_model<span class="token punctuation">,</span> input_shape<span class="token operator">=</span><span class="token punctuation">(</span>time_steps<span class="token punctuation">,</span> num_features<span class="token punctuation">,</span> <span class="token number">1</span><span class="token punctuation">)</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
combined_model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>LSTM<span class="token punctuation">(</span><span class="token number">64</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
combined_model<span class="token punctuation">.</span>add<span class="token punctuation">(</span>layers<span class="token punctuation">.</span>Dense<span class="token punctuation">(</span><span class="token number">1</span><span class="token punctuation">,</span> activation<span class="token operator">=</span><span class="token string">'sigmoid'</span><span class="token punctuation">)</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code creates a combined model that first applies the CNN model using <code>TimeDistributed</code> layer to process each time step separately. Then, it adds an LSTM layer to capture the sequential dependencies across time steps. Finally, it adds an output Dense layer with a sigmoid activation function for binary classification.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Compile the combined model</span>
combined_model<span class="token punctuation">.</span><span class="token builtin">compile</span><span class="token punctuation">(</span>optimizer<span class="token operator">=</span><span class="token string">'adam'</span><span class="token punctuation">,</span> loss<span class="token operator">=</span><span class="token string">'binary_crossentropy'</span><span class="token punctuation">,</span> metrics<span class="token operator">=</span><span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code compiles the combined model using the ‘adam’ optimizer, ‘binary_crossentropy’ loss function (suitable for binary classification), and ‘accuracy’ metric for model training.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Print the model summary</span>
combined_model<span class="token punctuation">.</span>summary<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code prints a summary of the combined model, including the layer configurations and the number of parameters.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Reshape the input data for LSTM</span>
X_train_lstm <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_train<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> time_steps<span class="token punctuation">,</span> num_features<span class="token punctuation">)</span>
X_test_lstm <span class="token operator">=</span> np<span class="token punctuation">.</span>array<span class="token punctuation">(</span>X_test<span class="token punctuation">)</span><span class="token punctuation">.</span>reshape<span class="token punctuation">(</span><span class="token operator">-</span><span class="token number">1</span><span class="token punctuation">,</span> time_steps<span class="token punctuation">,</span> num_features<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code reshapes the training and testing input features (X_train and X_test) for LSTM processing. For LSTM, the input data needs to have a 3D shape (batch_size, time_steps, num_features). In this case, the batch_size is set to -1 (automatically determined based on the data size), time_steps is set to 1 (as each month is considered a single time step), and num_features is the number of input features.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Train the combined model using the training data</span>
history <span class="token operator">=</span> combined_model<span class="token punctuation">.</span>fit<span class="token punctuation">(</span>X_train_lstm<span class="token punctuation">,</span> y_train<span class="token punctuation">,</span> epochs<span class="token operator">=</span><span class="token number">18</span><span class="token punctuation">,</span> batch_size<span class="token operator">=</span><span class="token number">16</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code trains the combined model using the training data (X_train_lstm and y_train) with 18 epochs and a batch size of 16. The training history is stored in the ‘history’ variable.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Evaluate the model on the test data</span>
test_loss<span class="token punctuation">,</span> test_accuracy <span class="token operator">=</span> combined_model<span class="token punctuation">.</span>evaluate<span class="token punctuation">(</span>X_test_lstm<span class="token punctuation">,</span> y_test<span class="token punctuation">,</span> verbose<span class="token operator">=</span><span class="token number">0</span><span class="token punctuation">)</span>
<span class="token keyword">print</span><span class="token punctuation">(</span><span class="token string">"Test Accuracy:"</span><span class="token punctuation">,</span> test_accuracy<span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code evaluates the combined model on the test data after training using the <code>evaluate</code> method. The accuracy and loss values are printed.</li>
</ul>
<pre class=" language-python"><code class="prism  language-python"><span class="token comment"># Plot the accuracy and loss over epochs</span>
<span class="token keyword">import</span> matplotlib<span class="token punctuation">.</span>pyplot <span class="token keyword">as</span> plt

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'accuracy'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Accuracy'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>

plt<span class="token punctuation">.</span>plot<span class="token punctuation">(</span>history<span class="token punctuation">.</span>history<span class="token punctuation">[</span><span class="token string">'loss'</span><span class="token punctuation">]</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>title<span class="token punctuation">(</span><span class="token string">'Model Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>xlabel<span class="token punctuation">(</span><span class="token string">'Epoch'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>ylabel<span class="token punctuation">(</span><span class="token string">'Loss'</span><span class="token punctuation">)</span>
plt<span class="token punctuation">.</span>show<span class="token punctuation">(</span><span class="token punctuation">)</span>
</code></pre>
<ul>
<li>The code uses matplotlib to plot the accuracy and loss over epochs to visualize the model’s performance during training. Two separate plots are generated for accuracy and loss, showing the trend of these metrics over the training epochs.</li>
</ul>

