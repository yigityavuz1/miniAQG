# Transcript for https://www.youtube.com/watch?v=9-Jl0dxWQs8


Kind: captions

Language: en

If you feed a large language model the phrase, Michael Jordan plays the sport of blank,

and you have it predict what comes next, and it correctly predicts basketball,

this would suggest that somewhere, inside its hundreds of billions of parameters,

it's baked in knowledge about a specific person and his specific sport.

And I think in general, anyone who's played around with one of these

models has the clear sense that it's memorized tons and tons of facts.

So a reasonable question you could ask is, how exactly does that work?

And where do those facts live?

Last December, a few researchers from Google DeepMind posted about work on this question,

and they were using this specific example of matching athletes to their sports.

And although a full mechanistic understanding of how facts are stored remains unsolved,

they had some interesting partial results, including the very general high-level

conclusion that the facts seem to live inside a specific part of these networks,

known fancifully as the multi-layer perceptrons, or MLPs for short.

In the last couple of chapters, you and I have been digging into

the details behind transformers, the architecture underlying large language models,

and also underlying a lot of other modern AI.

In the most recent chapter, we were focusing on a piece called Attention.

And the next step for you and me is to dig into the details of what happens inside

these multi-layer perceptrons, which make up the other big portion of the network.

The computation here is actually relatively simple,

especially when you compare it to attention.

It boils down essentially to a pair of matrix

multiplications with a simple something in between.

However, interpreting what these computations are doing is exceedingly challenging.

Our main goal here is to step through the computations and make them memorable,

but I'd like to do it in the context of showing a specific example of how

one of these blocks could, at least in principle, store a concrete fact.

Specifically, it'll be storing the fact that Michael Jordan plays basketball.

I should mention the layout here is inspired by a conversation

I had with one of those DeepMind researchers, Neil Nanda.

For the most part, I will assume that you've either watched the last two chapters,

or otherwise you have a basic sense for what a transformer is,

but refreshers never hurt, so here's the quick reminder of the overall flow.

You and I have been studying a model that's trained

to take in a piece of text and predict what comes next.

That input text is first broken into a bunch of tokens,

which means little chunks that are typically words or little pieces of words,

and each token is associated with a high-dimensional vector,

which is to say a long list of numbers.

This sequence of vectors then repeatedly passes through two kinds of operation,

attention, which allows the vectors to pass information between one another,

and then the multilayer perceptrons, the thing that we're gonna dig into today,

and also there's a certain normalization step in between.

After the sequence of vectors has flowed through many,

many different iterations of both of these blocks, by the end,

the hope is that each vector has soaked up enough information, both from the context,

all of the other words in the input, and also from the general knowledge that

was baked into the model weights through training,

that it can be used to make a prediction of what token comes next.

One of the key ideas that I want you to have in your mind is that all of

these vectors live in a very, very high-dimensional space,

and when you think about that space, different directions can encode different

kinds of meaning.

So a very classic example that I like to refer back to is how if you look

at the embedding of woman and subtract the embedding of man,

and you take that little step and you add it to another masculine noun,

something like uncle, you land somewhere very,

very close to the corresponding feminine noun.

In this sense, this particular direction encodes gender information.

The idea is that many other distinct directions in this super high-dimensional

space could correspond to other features that the model might want to represent.

In a transformer, these vectors don't merely encode the meaning of a single word, though.

As they flow through the network, they imbibe a much richer meaning based

on all the context around them, and also based on the model's knowledge.

Ultimately, each one needs to encode something far,

far beyond the meaning of a single word, since it needs to be sufficient to

predict what will come next.

We've already seen how attention blocks let you incorporate context,

but a majority of the model parameters actually live inside the MLP blocks,

and one thought for what they might be doing is that they offer extra capacity

to store facts.

Like I said, the lesson here is gonna center on the concrete toy example

of how exactly it could store the fact that Michael Jordan plays basketball.

Now, this toy example is gonna require that you and I make

a couple of assumptions about that high-dimensional space.

First, we'll suppose that one of the directions represents the idea of a first name

Michael, and then another nearly perpendicular direction represents the idea of the

last name Jordan, and then yet a third direction will represent the idea of basketball.

So specifically, what I mean by this is if you look in the network and

you pluck out one of the vectors being processed,

if its dot product with this first name Michael direction is one,

that's what it would mean for the vector to be encoding the idea of a

person with that first name.

Otherwise, that dot product would be zero or negative,

meaning the vector doesn't really align with that direction.

And for simplicity, let's completely ignore the very reasonable

question of what it might mean if that dot product was bigger than one.

Similarly, its dot product with these other directions would

tell you whether it represents the last name Jordan or basketball.

So let's say a vector is meant to represent the full name, Michael Jordan,

then its dot product with both of these directions would have to be one.

Since the text Michael Jordan spans two different tokens,

this would also mean we have to assume that an earlier attention block has successfully

passed information to the second of these two vectors so as to ensure that it can

encode both names.

With all of those as the assumptions, let's now dive into the meat of the lesson.

What happens inside a multilayer perceptron?

You might think of this sequence of vectors flowing into the block, and remember,

each vector was originally associated with one of the tokens from the input text.

What's gonna happen is that each individual vector from that sequence

goes through a short series of operations, we'll unpack them in just a moment,

and at the end, we'll get another vector with the same dimension.

That other vector is gonna get added to the original one that flowed in,

and that sum is the result flowing out.

This sequence of operations is something you apply to every vector in the sequence,

associated with every token in the input, and it all happens in parallel.

In particular, the vectors don't talk to each other in this step,

they're all kind of doing their own thing.

And for you and me, that actually makes it a lot simpler,

because it means if we understand what happens to just one of the

vectors through this block, we effectively understand what happens to all of them.

When I say this block is gonna encode the fact that Michael Jordan plays basketball,

what I mean is that if a vector flows in that encodes first name Michael and last

name Jordan, then this sequence of computations will produce something that includes

that direction basketball, which is what will add on to the vector in that position.

The first step of this process looks like multiplying that vector by a very big matrix.

No surprises there, this is deep learning.

And this matrix, like all of the other ones we've seen,

is filled with model parameters that are learned from data,

which you might think of as a bunch of knobs and dials that get tweaked and

tuned to determine what the model behavior is.

Now, one nice way to think about matrix multiplication is to imagine each row of

that matrix as being its own vector, and taking a bunch of dot products between

those rows and the vector being processed, which I'll label as E for embedding.

For example, suppose that very first row happened to equal

this first name Michael direction that we're presuming exists.

That would mean that the first component in this output, this dot product right here,

would be one if that vector encodes the first name Michael,

and zero or negative otherwise.

Even more fun, take a moment to think about what it would mean if that

first row was this first name Michael plus last name Jordan direction.

And for simplicity, let me go ahead and write that down as M plus J.

Then, taking a dot product with this embedding E,

things distribute really nicely, so it looks like M dot E plus J dot E.

And notice how that means the ultimate value would be two if the vector encodes the

full name Michael Jordan, and otherwise it would be one or something smaller than one.

And that's just one row in this matrix.

You might think of all of the other rows as in parallel asking some other kinds of

questions, probing at some other sorts of features of the vector being processed.

Very often this step also involves adding another vector to the output,

which is full of model parameters learned from data.

This other vector is known as the bias.

For our example, I want you to imagine that the value of this

bias in that very first component is negative one,

meaning our final output looks like that relevant dot product, but minus one.

You might very reasonably ask why I would want you to assume that the

model has learned this, and in a moment you'll see why it's very clean

and nice if we have a value here which is positive if and only if a vector

encodes the full name Michael Jordan, and otherwise it's zero or negative.

The total number of rows in this matrix, which is something

like the number of questions being asked, in the case of GPT-3,

whose numbers we've been following, is just under 50,000.

In fact, it's exactly four times the number of dimensions in this embedding space.

That's a design choice.

You could make it more, you could make it less,

but having a clean multiple tends to be friendly for hardware.

Since this matrix full of weights maps us into a higher dimensional space,

I'm gonna give it the shorthand W up.

I'll continue labeling the vector we're processing as E,

and let's label this bias vector as B up and put that all back down in the diagram.

At this point, a problem is that this operation is purely linear,

but language is a very non-linear process.

If the entry that we're measuring is high for Michael plus Jordan,

it would also necessarily be somewhat triggered by Michael plus Phelps

and also Alexis plus Jordan, despite those being unrelated conceptually.

What you really want is a simple yes or no for the full name.

So the next step is to pass this large intermediate

vector through a very simple non-linear function.

A common choice is one that takes all of the negative values and

maps them to zero and leaves all of the positive values unchanged.

And continuing with the deep learning tradition of overly fancy names,

this very simple function is often called the rectified linear unit, or ReLU for short.

Here's what the graph looks like.

So taking our imagined example where this first entry of the intermediate vector is one,

if and only if the full name is Michael Jordan and zero or negative otherwise,

after you pass it through the ReLU, you end up with a very clean value where

all of the zero and negative values just get clipped to zero.

So this output would be one for the full name Michael Jordan and zero otherwise.

In other words, it very directly mimics the behavior of an AND gate.

Often models will use a slightly modified function that's called the GELU,

which has the same basic shape, it's just a bit smoother.

But for our purposes, it's a little bit cleaner if we only think about the ReLU.

Also, when you hear people refer to the neurons of a transformer,

they're talking about these values right here.

Whenever you see that common neural network picture with a layer of dots and a

bunch of lines connecting to the previous layer, which we had earlier in this series,

that's typically meant to convey this combination of a linear step,

a matrix multiplication, followed by some simple term-wise nonlinear function like a ReLU.

You would say that this neuron is active whenever this value

is positive and that it's inactive if that value is zero.

The next step looks very similar to the first one.

You multiply by a very large matrix and you add on a certain bias term.

In this case, the number of dimensions in the output is back down to the size of

that embedding space, so I'm gonna go ahead and call this the down projection matrix.

And this time, instead of thinking of things row by row,

it's actually nicer to think of it column by column.

You see, another way that you can hold matrix multiplication in your head is to

imagine taking each column of the matrix and multiplying it by the corresponding

term in the vector that it's processing and adding together all of those rescaled columns.

The reason it's nicer to think about this way is because here the columns have the same

dimension as the embedding space, so we can think of them as directions in that space.

For instance, we will imagine that the model has learned to make that

first column into this basketball direction that we suppose exists.

What that would mean is that when the relevant neuron in that first position is active,

we'll be adding this column to the final result.

But if that neuron was inactive, if that number was zero, then this would have no effect.

And it doesn't just have to be basketball.

The model could also bake into this column and many other features that

it wants to associate with something that has the full name Michael Jordan.

And at the same time, all of the other columns in this matrix are telling you

what will be added to the final result if the corresponding neuron is active.

And if you have a bias in this case, it's something that you're

just adding every single time, regardless of the neuron values.

You might wonder what's that doing.

As with all parameter-filled objects here, it's kind of hard to say exactly.

Maybe there's some bookkeeping that the network needs to do,

but you can feel free to ignore it for now.

Making our notation a little more compact again,

I'll call this big matrix W down and similarly call that bias vector B down and

put that back into our diagram.

Like I previewed earlier, what you do with this final result is add it to the vector

that flowed into the block at that position and that gets you this final result.

So for example, if the vector flowing in encoded both first name Michael and last name

Jordan, then because this sequence of operations will trigger that AND gate,

it will add on the basketball direction, so what pops out will encode all of those

together.

And remember, this is a process happening to every one of those vectors in parallel.

In particular, taking the GPT-3 numbers, it means that this block doesn't just

have 50,000 neurons in it, it has 50,000 times the number of tokens in the input.

So that is the entire operation, two matrix products,

each with a bias added and a simple clipping function in between.

Any of you who watched the earlier videos of the series will recognize this

structure as the most basic kind of neural network that we studied there.

In that example, it was trained to recognize handwritten digits.

Over here, in the context of a transformer for a large language model,

this is one piece in a larger architecture and any attempt to interpret

what exactly it's doing is heavily intertwined with the idea of encoding

information into vectors of a high-dimensional embedding space.

That is the core lesson, but I do wanna step back and reflect on two different things,

the first of which is a kind of bookkeeping, and the second of which

involves a very thought-provoking fact about higher dimensions that

I actually didn't know until I dug into transformers.

In the last two chapters, you and I started counting up the total number of parameters

in GPT-3 and seeing exactly where they live, so let's quickly finish up the game here.

I already mentioned how this up projection matrix has just under 50,000 rows and

that each row matches the size of the embedding space, which for GPT-3 is 12,288.

Multiplying those together, it gives us 604 million parameters just for that matrix,

and the down projection has the same number of parameters just with a transposed shape.

So together, they give about 1.2 billion parameters.

The bias vector also accounts for a couple more parameters,

but it's a trivial proportion of the total, so I'm not even gonna show it.

In GPT-3, this sequence of embedding vectors flows through not one,

but 96 distinct MLPs, so the total number of parameters devoted

to all of these blocks adds up to about 116 billion.

This is around 2 thirds of the total parameters in the network,

and when you add it to everything that we had before, for the attention blocks,

the embedding, and the unembedding, you do indeed get that grand total of 175

billion as advertised.

It's probably worth mentioning there's another set of parameters associated

with those normalization steps that this explanation has skipped over,

but like the bias vector, they account for a very trivial proportion of the total.

As to that second point of reflection, you might be wondering if

this central toy example we've been spending so much time on

reflects how facts are actually stored in real large language models.

It is true that the rows of that first matrix can be thought of as

directions in this embedding space, and that means the activation of each

neuron tells you how much a given vector aligns with some specific direction.

It's also true that the columns of that second matrix tell

you what will be added to the result if that neuron is active.

Both of those are just mathematical facts.

However, the evidence does suggest that individual neurons very rarely

represent a single clean feature like Michael Jordan,

and there may actually be a very good reason this is the case,

related to an idea floating around interpretability researchers these

days known as superposition.

This is a hypothesis that might help to explain both why the models are

especially hard to interpret and also why they scale surprisingly well.

The basic idea is that if you have an n-dimensional space and you wanna

represent a bunch of different features using directions that are all

perpendicular to one another in that space, you know,

that way if you add a component in one direction,

it doesn't influence any of the other directions,

then the maximum number of vectors you can fit is only n, the number of dimensions.

To a mathematician, actually, this is the definition of dimension.

But where it gets interesting is if you relax that

constraint a little bit and you tolerate some noise.

Say you allow those features to be represented by vectors that aren't exactly

perpendicular, they're just nearly perpendicular, maybe between 89 and 91 degrees apart.

If we were in two or three dimensions, this makes no difference.

That gives you hardly any extra wiggle room to fit more vectors in,

which makes it all the more counterintuitive that for higher dimensions,

the answer changes dramatically.

I can give you a really quick and dirty illustration of this using some

scrappy Python that's going to create a list of 100-dimensional vectors,

each one initialized randomly, and this list is going to contain 10,000 distinct vectors,

so 100 times as many vectors as there are dimensions.

This plot right here shows the distribution of angles between pairs of these vectors.

So because they started at random, those angles could be anything from 0 to 180 degrees,

but you'll notice that already, even just for random vectors,

there's this heavy bias for things to be closer to 90 degrees.

Then what I'm going to do is run a certain optimization process that iteratively nudges

all of these vectors so that they try to become more perpendicular to one another.

After repeating this many different times, here's

what the distribution of angles looks like.

We have to actually zoom in on it here because all of the possible angles

between pairs of vectors sit inside this narrow range between 89 and 91 degrees.

In general, a consequence of something known as the Johnson-Lindenstrauss

lemma is that the number of vectors you can cram into a space that are nearly

perpendicular like this grows exponentially with the number of dimensions.

This is very significant for large language models,

which might benefit from associating independent ideas with nearly

perpendicular directions.

It means that it's possible for it to store many,

many more ideas than there are dimensions in the space that it's allotted.

This might partially explain why model performance seems to scale so well with size.

A space that has 10 times as many dimensions can store way,

way more than 10 times as many independent ideas.

And this is relevant not just to that embedding space where the vectors

flowing through the model live, but also to that vector full of neurons

in the middle of that multilayer perceptron that we just studied.

That is to say, at the sizes of GPT-3, it might not just be probing at 50,000 features,

but if it instead leveraged this enormous added capacity by using

nearly perpendicular directions of the space, it could be probing at many,

many more features of the vector being processed.

But if it was doing that, what it means is that individual

features aren't gonna be visible as a single neuron lighting up.

It would have to look like some specific combination of neurons instead, a superposition.

For any of you curious to learn more, a key relevant search term here is sparse

autoencoder, which is a tool that some of the interpretability people use to try to

extract what the true features are, even if they're very superimposed on all these

neurons.

I'll link to a couple really great anthropic posts all about this.

At this point, we haven't touched every detail of a transformer,

but you and I have hit the most important points.

The main thing that I wanna cover in a next chapter is the training process.

On the one hand, the short answer for how training works is that it's all

backpropagation, and we covered backpropagation in a separate context with earlier

chapters in the series.

But there is more to discuss, like the specific cost function used for language models,

the idea of fine-tuning using reinforcement learning with human feedback,

and the notion of scaling laws.

Quick note for the active followers among you,

there are a number of non-machine learning-related videos that I'm excited to

sink my teeth into before I make that next chapter, so it might be a while,

but I do promise it'll come in due time.

Thank you.