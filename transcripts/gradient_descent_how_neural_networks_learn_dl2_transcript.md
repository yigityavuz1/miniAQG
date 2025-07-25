# Transcript for https://www.youtube.com/watch?v=IHZwWFHWa-w


Kind: captions

Language: en

Last video I laid out the structure of a neural network.

I'll give a quick recap here so that it's fresh in our minds,

and then I have two main goals for this video.

The first is to introduce the idea of gradient descent,

which underlies not only how neural networks learn,

but how a lot of other machine learning works as well.

Then after that we'll dig in a little more into how this particular network performs,

and what those hidden layers of neurons end up looking for.

As a reminder, our goal here is the classic example of handwritten digit recognition,

the hello world of neural networks.

These digits are rendered on a 28x28 pixel grid,

each pixel with some grayscale value between 0 and 1.

Those are what determine the activations of 784 neurons in the input layer of the network.

And then the activation for each neuron in the following layers is based on a weighted

sum of all the activations in the previous layer, plus some special number called a bias.

Then you compose that sum with some other function,

like the sigmoid squishification, or a relu, the way I walked through last video.

In total, given the somewhat arbitrary choice of two hidden layers with 16 neurons each,

the network has about 13,000 weights and biases that we can adjust,

and it's these values that determine what exactly the network actually does.

Then what we mean when we say that this network classifies a given digit is that

the brightest of those 10 neurons in the final layer corresponds to that digit.

And remember, the motivation we had in mind here for the layered structure

was that maybe the second layer could pick up on the edges,

and the third layer might pick up on patterns like loops and lines,

and the last one could just piece together those patterns to recognize digits.

So here, we learn how the network learns.

What we want is an algorithm where you can show this network a whole bunch of

training data, which comes in the form of a bunch of different images of handwritten

digits, along with labels for what they're supposed to be,

and it'll adjust those 13,000 weights and biases so as to improve its performance

on the training data.

Hopefully, this layered structure will mean that what it

learns generalizes to images beyond that training data.

The way we test that is that after you train the network,

you show it more labeled data that it's never seen before,

and you see how accurately it classifies those new images.

Fortunately for us, and what makes this such a common example to start with,

is that the good people behind the MNIST database have put together a collection of tens

of thousands of handwritten digit images, each one labeled with the numbers they're

supposed to be.

And as provocative as it is to describe a machine as learning,

once you see how it works, it feels a lot less like some crazy sci-fi premise,

and a lot more like a calculus exercise.

I mean, basically it comes down to finding the minimum of a certain function.

Remember, conceptually, we're thinking of each neuron as being connected to all

the neurons in the previous layer, and the weights in the weighted sum defining

its activation are kind of like the strengths of those connections,

and the bias is some indication of whether that neuron tends to be active or inactive.

And to start things off, we're just going to initialize

all of those weights and biases totally randomly.

Needless to say, this network is going to perform pretty horribly on

a given training example, since it's just doing something random.

For example, you feed in this image of a 3, and the output layer just looks like a mess.

So what you do is define a cost function, a way of telling the computer,

no, bad computer, that output should have activations which are 0 for most neurons,

but 1 for this neuron, what you gave me is utter trash.

To say that a little more mathematically, you add up the squares of the differences

between each of those trash output activations and the value you want them to have,

and this is what we'll call the cost of a single training example.

Notice this sum is small when the network confidently classifies the image correctly,

but it's large when the network seems like it doesn't know what it's doing.

So then what you do is consider the average cost over all of

the tens of thousands of training examples at your disposal.

This average cost is our measure for how lousy the network is,

and how bad the computer should feel.

And that's a complicated thing.

Remember how the network itself was basically a function,

one that takes in 784 numbers as inputs, the pixel values,

and spits out 10 numbers as its output, and in a sense it's parameterized

by all these weights and biases?

Well the cost function is a layer of complexity on top of that.

It takes as its input those 13,000 or so weights and biases,

and spits out a single number describing how bad those weights and biases are,

and the way it's defined depends on the network's behavior over all the tens of

thousands of pieces of training data.

That's a lot to think about.

But just telling the computer what a crappy job it's doing isn't very helpful.

You want to tell it how to change those weights and biases so that it gets better.

To make it easier, rather than struggling to imagine a function with 13,000 inputs,

just imagine a simple function that has one number as an input and one number as an

output.

How do you find an input that minimizes the value of this function?

Calculus students will know that you can sometimes figure out that minimum explicitly,

but that's not always feasible for really complicated functions,

certainly not in the 13,000 input version of this situation for our crazy complicated

neural network cost function.

A more flexible tactic is to start at any input,

and figure out which direction you should step to make that output lower.

Specifically, if you can figure out the slope of the function where you are,

then shift to the left if that slope is positive,

and shift the input to the right if that slope is negative.

If you do this repeatedly, at each point checking the new slope and taking the

appropriate step, you're going to approach some local minimum of the function.

The image you might have in mind here is a ball rolling down a hill.

Notice, even for this really simplified single input function,

there are many possible valleys that you might land in,

depending on which random input you start at,

and there's no guarantee that the local minimum you land in is going to

be the smallest possible value of the cost function.

That will carry over to our neural network case as well.

And I also want you to notice how if you make your step sizes proportional to the slope,

then when the slope is flattening out towards the minimum,

your steps get smaller and smaller, and that kind of helps you from overshooting.

Bumping up the complexity a bit, imagine instead

a function with two inputs and one output.

You might think of the input space as the xy-plane,

and the cost function as being graphed as a surface above it.

Now instead of asking about the slope of the function,

you have to ask which direction you should step in this input

space so as to decrease the output of the function most quickly.

In other words, what's the downhill direction?

Again, it's helpful to think of a ball rolling down that hill.

Those of you familiar with multivariable calculus will know that the

gradient of a function gives you the direction of steepest ascent,

which direction should you step to increase the function most quickly.

Naturally enough, taking the negative of that gradient gives you

the direction to step that decreases the function most quickly.

Even more than that, the length of this gradient vector is

an indication for just how steep that steepest slope is.

If you're unfamiliar with multivariable calculus and want to learn more,

check out some of the work I did for Khan Academy on the topic.

Honestly though, all that matters for you and me right now is that

in principle there exists a way to compute this vector,

this vector that tells you what the downhill direction is and how steep it is.

You'll be okay if that's all you know and you're not rock solid on the details.

Cause If you can get that, the algorithm for minimizing the function is to compute this

gradient direction, then take a small step downhill, and repeat that over and over.

It's the same basic idea for a function that has 13,000 inputs instead of 2 inputs.

Imagine organizing all 13,000 weights and biases

of our network into a giant column vector.

The negative gradient of the cost function is just a vector,

it's some direction inside this insanely huge input space that tells you which

nudges to all of those numbers is going to cause the most rapid decrease to

the cost function.

And of course, with our specially designed cost function,

changing the weights and biases to decrease it means making the

output of the network on each piece of training data look less like

a random array of 10 values, and more like an actual decision we want it to make.

It's important to remember, this cost function involves an average over all of the

training data, so if you minimize it, it means it's a better performance on all of those

samples.

The algorithm for computing this gradient efficiently,

which is effectively the heart of how a neural network learns,

is called backpropagation, and it's what I'm going to be talking about next video.

There, I really want to take the time to walk through what exactly happens to

each weight and bias for a given piece of training data,

trying to give an intuitive feel for what's happening beyond the pile of relevant

calculus and formulas.

Right here, right now, the main thing I want you to know,

independent of implementation details, is that what we mean when we

talk about a network learning is that it's just minimizing a cost function.

And notice, one consequence of that is that it's important for this cost function to have

a nice smooth output, so that we can find a local minimum by taking little steps

downhill.

This is why, by the way, artificial neurons have continuously ranging activations,

rather than simply being active or inactive in a binary way,

the way biological neurons are.

This process of repeatedly nudging an input of a function by some

multiple of the negative gradient is called gradient descent.

It's a way to converge towards some local minimum of a cost function,

basically a valley in this graph.

I'm still showing the picture of a function with two inputs, of course,

because nudges in a 13,000 dimensional input space are a little hard to

wrap your mind around, but there is a nice non-spatial way to think about this.

Each component of the negative gradient tells us two things.

The sign, of course, tells us whether the corresponding

component of the input vector should be nudged up or down.

But importantly, the relative magnitudes of all these

components kind of tells you which changes matter more.

You see, in our network, an adjustment to one of the weights might have a much

greater impact on the cost function than the adjustment to some other weight.

Some of these connections just matter more for our training data.

So a way you can think about this gradient vector of our mind-warpingly massive

cost function is that it encodes the relative importance of each weight and bias,

that is, which of these changes is going to carry the most bang for your buck.

This really is just another way of thinking about direction.

To take a simpler example, if you have some function with two variables as an input,

and you compute that its gradient at some particular point comes out as 3,1,

then on the one hand you can interpret that as saying that when you're

standing at that input, moving along this direction increases the function most quickly,

that when you graph the function above the plane of input points,

that vector is what's giving you the straight uphill direction.

But another way to read that is to say that changes to this first variable have 3

times the importance as changes to the second variable,

that at least in the neighborhood of the relevant input,

nudging the x-value carries a lot more bang for your buck.

Let's zoom out and sum up where we are so far.

The network itself is this function with 784 inputs and 10 outputs,

defined in terms of all these weighted sums.

The cost function is a layer of complexity on top of that.

It takes the 13,000 weights and biases as inputs and spits out

a single measure of lousiness based on the training examples.

And the gradient of the cost function is one more layer of complexity still.

It tells us what nudges to all these weights and biases cause the

fastest change to the value of the cost function,

which you might interpret as saying which changes to which weights matter the most.

So, when you initialize the network with random weights and biases,

and adjust them many times based on this gradient descent process,

how well does it actually perform on images it's never seen before?

The one I've described here, with the two hidden layers of 16 neurons each,

chosen mostly for aesthetic reasons, is not bad,

classifying about 96% of the new images it sees correctly.

And honestly, if you look at some of the examples it messes up on,

you feel compelled to cut it a little slack.

Now if you play around with the hidden layer structure and make a couple tweaks,

you can get this up to 98%.

And that's pretty good!

It's not the best, you can certainly get better performance by getting more sophisticated

than this plain vanilla network, but given how daunting the initial task is,

I think there's something incredible about any network doing this well on images it's

never seen before, given that we never specifically told it what patterns to look for.

Originally, the way I motivated this structure was by describing a hope we might have,

that the second layer might pick up on little edges,

that the third layer would piece together those edges to recognize loops

and longer lines, and that those might be pieced together to recognize digits.

So is this what our network is actually doing?

Well, for this one at least, not at all.

Remember how last video we looked at how the weights of the connections from all

the neurons in the first layer to a given neuron in the second layer can be

visualized as a given pixel pattern that the second layer neuron is picking up on?

Well, when we actually do that for the weights associated with these transitions,

from the first layer to the next, instead of picking up on isolated little edges here

and there, they look, well, almost random, just with some very loose patterns in the

middle there.

It would seem that in the unfathomably large 13,000 dimensional space

of possible weights and biases, our network found itself a happy

little local minimum that, despite successfully classifying most images,

doesn't exactly pick up on the patterns we might have hoped for.

And to really drive this point home, watch what happens when you input a random image.

If the system was smart, you might expect it to feel uncertain,

maybe not really activating any of those 10 output neurons or activating them

all evenly, but instead it confidently gives you some nonsense answer,

as if it feels as sure that this random noise is a 5 as it does that an actual

image of a 5 is a 5.

Phrased differently, even if this network can recognize digits pretty well,

it has no idea how to draw them.

A lot of this is because it's such a tightly constrained training setup.

I mean, put yourself in the network's shoes here.

From its point of view, the entire universe consists of nothing but clearly

defined unmoving digits centered in a tiny grid,

and its cost function never gave it any incentive to be anything but utterly

confident in its decisions.

So with this as the image of what those second layer neurons are really doing,

you might wonder why I would introduce this network with the

motivation of picking up on edges and patterns.

I mean, that's just not at all what it ends up doing.

Well, this is not meant to be our end goal, but instead a starting point.

Frankly, this is old technology, the kind researched in the 80s and 90s,

and you do need to understand it before you can understand more detailed modern

variants, and it clearly is capable of solving some interesting problems,

but the more you dig into what those hidden layers are really doing,

the less intelligent it seems.

Shifting the focus for a moment from how networks learn to how you learn,

that'll only happen if you engage actively with the material here somehow.

One pretty simple thing I want you to do is just pause right now and think deeply

for a moment about what changes you might make to this system and how it perceives

images if you wanted it to better pick up on things like edges and patterns.

But better than that, to actually engage with the material,

I highly recommend the book by Michael Nielsen on deep learning and neural networks.

In it, you can find the code and the data to download and play with for this exact

example, and the book will walk you through step by step what that code is doing.

What's awesome is that this book is free and publicly available,

so if you do get something out of it, consider joining me in making a donation towards

Nielsen's efforts.

I've also linked a couple other resources I like a lot in the description,

including the phenomenal and beautiful blog post by Chris Ola and the articles in

Distill.

To close things off here for the last few minutes,

I want to jump back into a snippet of the interview I had with Leisha Lee.

You might remember her from the last video, she did her PhD work in deep learning.

In this little snippet she talks about two recent papers that really dig into

how some of the more modern image recognition networks are actually learning.

Just to set up where we were in the conversation,

the first paper took one of these particularly deep neural networks that's really good

at image recognition, and instead of training it on a properly labeled dataset,

shuffled all the labels around before training.

Obviously the testing accuracy here was going to be no better than random,

since everything's just randomly labeled. But it was still able to achieve

the same training accuracy as you would on a properly labeled dataset.

Basically, the millions of weights for this particular network were

enough for it to just memorize the random data,

which raises the question for whether minimizing this cost function

actually corresponds to any sort of structure in the image, or is it just memorization?

...to memorize the entire dataset of what the correct classification is.

And so half a year later at ICML this year, there was not exactly rebuttal paper,

but paper that addressed some aspects of like, hey,

actually these networks are doing something a little bit smarter than that.

If you look at that accuracy curve if you were just training on a random data set

that curve went down very slowly, almost in a linear fashion.

So you’re really struggling to find that local minimum of the right weights.

Whereas if you're actually training on a structured dataset,

one that has the right labels, you fiddle around a little bit in the beginning,

but then you kind of dropped very fast to get to that accuracy level,

and so in some sense it was easier to find that local maxima.

And so what was also interesting about that is it brings into light another paper from

actually a couple of years ago, which has a lot more simplifications about the network

layers, but one of the results was saying how if you look at the optimization landscape,

the local minima that these networks tend to learn are actually of equal quality,

so in some sense if your dataset is structured,

you should be able to find that much more easily.

My thanks, as always, to those of you supporting on Patreon.

I've said before just what a game changer Patreon is,

but these videos really would not be possible without you.

I also want to give a special thanks to the VC firm Amplify Partners

and their support of these initial videos in the series. Thank you.