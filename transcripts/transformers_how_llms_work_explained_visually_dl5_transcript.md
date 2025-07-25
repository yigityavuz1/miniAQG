# Transcript for https://www.youtube.com/watch?v=wjZofJX0v4M


Kind: captions

Language: en

The initials GPT stand for Generative Pretrained Transformer.

So that first word is straightforward enough, these are bots that generate new text.

Pretrained refers to how the model went through a process of learning

from a massive amount of data, and the prefix insinuates that there's

more room to fine-tune it on specific tasks with additional training.

But the last word, that's the real key piece.

A transformer is a specific kind of neural network, a machine learning model,

and it's the core invention underlying the current boom in AI.

What I want to do with this video and the following chapters is go through a

visually-driven explanation for what actually happens inside a transformer.

We're going to follow the data that flows through it and go step by step.

There are many different kinds of models that you can build using transformers.

Some models take in audio and produce a transcript.

This sentence comes from a model going the other way around,

producing synthetic speech just from text.

All those tools that took the world by storm in 2022 like DALL-E and Midjourney

that take in a text description and produce an image are based on transformers.

Even if I can't quite get it to understand what a pi creature is supposed to be,

I'm still blown away that this kind of thing is even remotely possible.

And the original transformer introduced in 2017 by Google was invented for

the specific use case of translating text from one language into another.

But the variant that you and I will focus on, which is the type that

underlies tools like ChatGPT, will be a model that's trained to take in a piece of text,

maybe even with some surrounding images or sound accompanying it,

and produce a prediction for what comes next in the passage.

That prediction takes the form of a probability distribution

over many different chunks of text that might follow.

At first glance, you might think that predicting the next word

feels like a very different goal from generating new text.

But once you have a prediction model like this,

a simple thing you could try to make it generate, a longer piece of text,

is to give it an initial snippet to work with,

have it take a random sample from the distribution it just generated,

append that sample to the text, and then run the whole process again to make

a new prediction based on all the new text, including what it just added.

I don't know about you, but it really doesn't feel like this should actually work.

In this animation, for example, I'm running GPT-2 on my laptop and having it repeatedly

predict and sample the next chunk of text to generate a story based on the seed text.

The story just doesn't actually really make that much sense.

But if I swap it out for API calls to GPT-3 instead, which is the same basic model,

just much bigger, suddenly almost magically we do get a sensible story,

one that even seems to infer that a pi creature would live in a land of math and

computation.

This process here of repeated prediction and sampling is essentially

what's happening when you interact with ChatGPT,

or any of these other large language models, and you see them producing

one word at a time.

In fact, one feature that I would very much enjoy is the ability to

see the underlying distribution for each new word that it chooses.

Let's kick things off with a very high level preview

of how data flows through a transformer.

We will spend much more time motivating and interpreting and expanding

on the details of each step, but in broad strokes,

when one of these chatbots generates a given word, here's what's going on under the hood.

First, the input is broken up into a bunch of little pieces.

These pieces are called tokens, and in the case of text these tend to be

words or little pieces of words or other common character combinations.

If images or sound are involved, then tokens could be little

patches of that image or little chunks of that sound.

Each one of these tokens is then associated with a vector, meaning some list of numbers,

which is meant to somehow encode the meaning of that piece.

If you think of these vectors as giving coordinates in some very high dimensional space,

words with similar meanings tend to land on vectors that are

close to each other in that space.

This sequence of vectors then passes through an operation that's

known as an attention block, and this allows the vectors to talk to

each other and pass information back and forth to update their values.

For example, the meaning of the word model in the phrase "a machine learning

model" is different from its meaning in the phrase "a fashion model".

The attention block is what's responsible for figuring out which

words in context are relevant to updating the meanings of which other words,

and how exactly those meanings should be updated.

And again, whenever I use the word meaning, this is

somehow entirely encoded in the entries of those vectors.

After that, these vectors pass through a different kind of operation,

and depending on the source that you're reading this will be referred

to as a multi-layer perceptron or maybe a feed-forward layer.

And here the vectors don't talk to each other,

they all go through the same operation in parallel.

And while this block is a little bit harder to interpret,

later on we'll talk about how the step is a little bit like asking a long list

of questions about each vector, and then updating them based on the answers

to those questions.

All of the operations in both of these blocks look like a

giant pile of matrix multiplications, and our primary job is

going to be to understand how to read the underlying matrices.

I'm glossing over some details about some normalization steps that happen in between,

but this is after all a high-level preview.

After that, the process essentially repeats, you go back and forth

between attention blocks and multi-layer perceptron blocks,

until at the very end the hope is that all of the essential meaning

of the passage has somehow been baked into the very last vector in the sequence.

We then perform a certain operation on that last vector that produces a probability

distribution over all possible tokens, all possible little chunks of text that might

come next.

And like I said, once you have a tool that predicts what comes next

given a snippet of text, you can feed it a little bit of seed text and

have it repeatedly play this game of predicting what comes next,

sampling from the distribution, appending it, and then repeating over and over.

Some of you in the know may remember how long before ChatGPT came into the scene,

this is what early demos of GPT-3 looked like,

you would have it autocomplete stories and essays based on an initial snippet.

To make a tool like this into a chatbot, the easiest starting point is to have a

little bit of text that establishes the setting of a user interacting with a

helpful AI assistant, what you would call the system prompt,

and then you would use the user's initial question or prompt as the first bit of

dialogue, and then you have it start predicting what such a helpful AI assistant

would say in response.

There is more to say about an added step of training that's required

to make this work well, but at a high level this is the idea.

In this chapter, you and I are going to expand on the details of what happens at the very

beginning of the network, at the very end of the network,

and I also want to spend a lot of time reviewing some important bits of background

knowledge, things that would have been second nature to any machine learning engineer by

the time transformers came around.

If you're comfortable with that background knowledge and a little impatient,

you could probably feel free to skip to the next chapter,

which is going to focus on the attention blocks,

generally considered the heart of the transformer.

After that, I want to talk more about these multi-layer perceptron blocks,

how training works, and a number of other details that will have been skipped up to

that point.

For broader context, these videos are additions to a mini-series about deep learning,

and it's okay if you haven't watched the previous ones,

I think you can do it out of order, but before diving into transformers specifically,

I do think it's worth making sure that we're on the same page about the basic premise

and structure of deep learning.

At the risk of stating the obvious, this is one approach to machine learning,

which describes any model where you're using data to somehow determine how a model

behaves.

What I mean by that is, let's say you want a function that takes in

an image and it produces a label describing it,

or our example of predicting the next word given a passage of text,

or any other task that seems to require some element of intuition

and pattern recognition.

We almost take this for granted these days, but the idea with machine learning is that

rather than trying to explicitly define a procedure for how to do that task in code,

which is what people would have done in the earliest days of AI,

instead you set up a very flexible structure with tunable parameters,

like a bunch of knobs and dials, and then, somehow,

you use many examples of what the output should look like for a given input to tweak

and tune the values of those parameters to mimic this behavior.

For example, maybe the simplest form of machine learning is linear regression,

where your inputs and outputs are each single numbers,

something like the square footage of a house and its price,

and what you want is to find a line of best fit through this data, you know,

to predict future house prices.

That line is described by two continuous parameters,

say the slope and the y-intercept, and the goal of linear

regression is to determine those parameters to closely match the data.

Needless to say, deep learning models get much more complicated.

GPT-3, for example, has not two, but 175 billion parameters.

But here's the thing, it's not a given that you can create some giant

model with a huge number of parameters without it either grossly

overfitting the training data or being completely intractable to train.

Deep learning describes a class of models that in the

last couple decades have proven to scale remarkably well.

What unifies them is that they all use the same training algorithm,

it's called backpropagation, we talked about it in previous chapters,

and the context that I want you to have as we go in is that in order for this

training algorithm to work well at scale, these models have to follow a certain

specific format.

And if you know this format going in, it helps to explain many of the choices for how a

transformer processes language, which otherwise run the risk of feeling kinda arbitrary.

First, whatever kind of model you're making, the

input has to be formatted as an array of real numbers.

This could simply mean a list of numbers, it could be a two-dimensional array,

or very often you deal with higher dimensional arrays,

where the general term used is tensor.

You often think of that input data as being progressively transformed into many

distinct layers, where again, each layer is always structured as some kind of

array of real numbers, until you get to a final layer which you consider the output.

For example, the final layer in our text processing model is a list of numbers

representing the probability distribution for all possible next tokens.

In deep learning, these model parameters are almost always referred to as weights,

and this is because a key feature of these models is that the only way these

parameters interact with the data being processed is through weighted sums.

You also sprinkle some non-linear functions throughout,

but they won't depend on parameters.

Typically, though, instead of seeing the weighted sums all naked

and written out explicitly like this, you'll instead find them

packaged together as various components in a matrix vector product.

It amounts to saying the same thing, if you think back to how matrix vector

multiplication works, each component in the output looks like a weighted sum.

It's just often conceptually cleaner for you and me to think

about matrices that are filled with tunable parameters that

transform vectors that are drawn from the data being processed.

For example, those 175 billion weights in GPT-3 are

organized into just under 28,000 distinct matrices.

Those matrices in turn fall into eight different categories,

and what you and I are going to do is step through each one of those categories to

understand what that type does.

As we go through, I think it's kind of fun to reference the specific

numbers from GPT-3 to count up exactly where those 175 billion come from.

Even if nowadays there are bigger and better models,

this one has a certain charm as the first large-language

model to really capture the world's attention outside of ML communities.

Also, practically speaking, companies tend to keep much tighter

lips around the specific numbers for more modern networks.

I just want to set the scene going in, that as you peek under the

hood to see what happens inside a tool like ChatGPT,

almost all of the actual computation looks like matrix vector multiplication.

There's a little bit of a risk getting lost in the sea of billions of numbers,

but you should draw a very sharp distinction in your mind between

the weights of the model, which I'll always color in blue or red,

and the data being processed, which I'll always color in gray.

The weights are the actual brains, they are the things learned during training,

and they determine how it behaves.

The data being processed simply encodes whatever specific input is

fed into the model for a given run, like an example snippet of text.

With all of that as foundation, let's dig into the first step of this text processing

example, which is to break up the input into little chunks and turn those chunks into

vectors.

I mentioned how those chunks are called tokens,

which might be pieces of words or punctuation,

but every now and then in this chapter and especially in the next one,

I'd like to just pretend that it's broken more cleanly into words.

Because we humans think in words, this will just make it much

easier to reference little examples and clarify each step.

The model has a predefined vocabulary, some list of all possible words,

say 50,000 of them, and the first matrix that we'll encounter,

known as the embedding matrix, has a single column for each one of these words.

These columns are what determines what vector each word turns into in that first step.

We label it W_E, and like all the matrices we see,

its values begin random, but they're going to be learned based on data.

Turning words into vectors was common practice in machine learning long before

transformers, but it's a little weird if you've never seen it before,

and it sets the foundation for everything that follows,

so let's take a moment to get familiar with it.

We often call this embedding a word, which invites you to think of these

vectors very geometrically as points in some high dimensional space.

Visualizing a list of three numbers as coordinates for points in 3D space would

be no problem, but word embeddings tend to be much much higher dimensional.

In GPT-3 they have 12,288 dimensions, and as you'll see,

it matters to work in a space that has a lot of distinct directions.

In the same way that you could take a two-dimensional slice through a 3D space

and project all the points onto that slice, for the sake of animating word

embeddings that a simple model is giving me, I'm going to do an analogous

thing by choosing a three-dimensional slice through this very high dimensional space,

and projecting the word vectors down onto that and displaying the results.

The big idea here is that as a model tweaks and tunes its weights to determine

how exactly words get embedded as vectors during training,

it tends to settle on a set of embeddings where directions in the space have a

kind of semantic meaning.

For the simple word-to-vector model I'm running here,

if I run a search for all the words whose embeddings are closest to that of tower,

you'll notice how they all seem to give very similar tower-ish vibes.

And if you want to pull up some Python and play along at home,

this is the specific model that I'm using to make the animations.

It's not a transformer, but it's enough to illustrate the

idea that directions in the space can carry semantic meaning.

A very classic example of this is how if you take the difference between

the vectors for woman and man, something you would visualize as a

little vector in the space connecting the tip of one to the tip of the other,

it's very similar to the difference between king and queen.

So let's say you didn't know the word for a female monarch,

you could find it by taking king, adding this woman minus man direction,

and searching for the embedding closest to that point.

At least, kind of.

Despite this being a classic example for the model I'm playing with,

the true embedding of queen is actually a little farther off than this would suggest,

presumably because the way queen is used in training data is not merely a feminine

version of king.

When I played around, family relations seemed to illustrate the idea much better.

The point is, it looks like during training the model found it advantageous to

choose embeddings such that one direction in this space encodes gender information.

Another example is that if you take the embedding of Italy,

and you subtract the embedding of Germany, and add that to the embedding of Hitler,

you get something very close to the embedding of Mussolini.

It's as if the model learned to associate some directions with Italian-ness,

and others with WWII axis leaders.

Maybe my favorite example in this vein is how in some models,

if you take the difference between Germany and Japan, and add it to sushi,

you end up very close to bratwurst.

Also in playing this game of finding nearest neighbors,

I was very pleased to see how close cat was to both beast and monster.

One bit of mathematical intuition that's helpful to have in mind,

especially for the next chapter, is how the dot product of two

vectors can be thought of as a way to measure how well they align.

Computationally, dot products involve multiplying all the

corresponding components and then adding the results, which is good,

since so much of our computation has to look like weighted sums.

Geometrically, the dot product is positive when vectors point in similar directions,

it's zero if they're perpendicular, and it's negative whenever

they point in opposite directions.

For example, let's say you were playing with this model,

and you hypothesize that the embedding of cats minus cat might represent a sort of

plurality direction in this space.

To test this, I'm going to take this vector and compute its dot

product against the embeddings of certain singular nouns,

and compare it to the dot products with the corresponding plural nouns.

If you play around with this, you'll notice that the plural ones

do indeed seem to consistently give higher values than the singular ones,

indicating that they align more with this direction.

It's also fun how if you take this dot product with the embeddings of the words one,

two, three, and so on, they give increasing values,

so it's as if we can quantitatively measure how plural the model finds a given word.

Again, the specifics for how words get embedded is learned using data.

This embedding matrix, whose columns tell us what happens to each word,

is the first pile of weights in our model.

Using the GPT-3 numbers, the vocabulary size specifically is 50,257,

and again, technically this consists not of words per se, but of tokens.

The embedding dimension is 12,288, and multiplying those

tells us this consists of about 617 million weights.

Let's go ahead and add this to a running tally,

remembering that by the end we should count up to 175 billion.

In the case of transformers, you really want to think of the vectors

in this embedding space as not merely representing individual words.

For one thing, they also encode information about the position of that word,

which we'll talk about later, but more importantly,

you should think of them as having the capacity to soak in context.

A vector that started its life as the embedding of the word king, for example,

might progressively get tugged and pulled by various blocks in this network,

so that by the end it points in a much more specific and nuanced direction that

somehow encodes that it was a king who lived in Scotland,

and who had achieved his post after murdering the previous king,

and who's being described in Shakespearean language.

Think about your own understanding of a given word.

The meaning of that word is clearly informed by the surroundings,

and sometimes this includes context from a long distance away,

so in putting together a model that has the ability to predict what word comes next,

the goal is to somehow empower it to incorporate context efficiently.

To be clear, in that very first step, when you create the array of

vectors based on the input text, each one of those is simply plucked

out of the embedding matrix, so initially each one can only encode

the meaning of a single word without any input from its surroundings.

But you should think of the primary goal of this network that it flows through

as being to enable each one of those vectors to soak up a meaning that's much

more rich and specific than what mere individual words could represent.

The network can only process a fixed number of vectors at a time,

known as its context size.

For GPT-3 it was trained with a context size of 2048,

so the data flowing through the network always looks like this array of 2048 columns,

each of which has 12,000 dimensions.

This context size limits how much text the transformer can

incorporate when it's making a prediction of the next word.

This is why long conversations with certain chatbots,

like the early versions of ChatGPT, often gave the feeling of

the bot kind of losing the thread of conversation as you continued too long.

We'll go into the details of attention in due time,

but skipping ahead I want to talk for a minute about what happens at the very end.

Remember, the desired output is a probability

distribution over all tokens that might come next.

For example, if the very last word is Professor,

and the context includes words like Harry Potter,

and immediately preceding we see least favorite teacher,

and also if you give me some leeway by letting me pretend that tokens simply

look like full words, then a well-trained network that had built up knowledge

of Harry Potter would presumably assign a high number to the word Snape.

This involves two different steps.

The first one is to use another matrix that maps the very last vector in that

context to a list of 50,000 values, one for each token in the vocabulary.

Then there's a function that normalizes this into a probability distribution,

it's called softmax and we'll talk more about it in just a second,

but before that it might seem a little bit weird to only use this last embedding

to make a prediction, when after all in that last step there are thousands of

other vectors in the layer just sitting there with their own context-rich meanings.

This has to do with the fact that in the training process it turns out to be

much more efficient if you use each one of those vectors in the final layer

to simultaneously make a prediction for what would come immediately after it.

There's a lot more to be said about training later on,

but I just want to call that out right now.

This matrix is called the Unembedding matrix and we give it the label WU.

Again, like all the weight matrices we see, its entries begin at random,

but they are learned during the training process.

Keeping score on our total parameter count, this Unembedding

matrix has one row for each word in the vocabulary,

and each row has the same number of elements as the embedding dimension.

It's very similar to the embedding matrix, just with the order swapped,

so it adds another 617 million parameters to the network,

meaning our count so far is a little over a billion,

a small but not wholly insignificant fraction of the 175 billion

we'll end up with in total.

As the very last mini-lesson for this chapter,

I want to talk more about this softmax function,

since it makes another appearance for us once we dive into the attention blocks.

The idea is that if you want a sequence of numbers to act as a probability distribution,

say a distribution over all possible next words,

then each value has to be between 0 and 1, and you also need all of them to add up to 1.

However, if you're playing the deep learning game where everything you do looks like

matrix-vector multiplication, the outputs you get by default don't abide by this at all.

The values are often negative, or much bigger than 1,

and they almost certainly don't add up to 1.

Softmax is the standard way to turn an arbitrary list of numbers

into a valid distribution in such a way that the largest values end up closest to 1,

and the smaller values end up very close to 0.

That's all you really need to know.

But if you're curious, the way it works is to first raise e to the power

of each of the numbers, which means you now have a list of positive values,

and then you can take the sum of all those positive values and divide each

term by that sum, which normalizes it into a list that adds up to 1.

You'll notice that if one of the numbers in the input is meaningfully bigger than the

rest, then in the output the corresponding term dominates the distribution,

so if you were sampling from it you'd almost certainly just be picking the maximizing

input.

But it's softer than just picking the max in the sense that when other values

are similarly large, they also get meaningful weight in the distribution,

and everything changes continuously as you continuously vary the inputs.

In some situations, like when ChatGPT is using this distribution to create a next word,

there's room for a little bit of extra fun by adding a little extra spice into this

function, with a constant T thrown into the denominator of those exponents.

We call it the temperature, since it vaguely resembles the role of temperature in

certain thermodynamics equations, and the effect is that when T is larger,

you give more weight to the lower values, meaning the distribution is a little bit

more uniform, and if T is smaller, then the bigger values will dominate more

aggressively, where in the extreme, setting T equal to zero means all of the weight

goes to maximum value.

For example, I'll have GPT-3 generate a story with the seed text,

"once upon a time there was A", but I'll use different temperatures in each case.

Temperature zero means that it always goes with the most predictable word,

and what you get ends up being a trite derivative of Goldilocks.

A higher temperature gives it a chance to choose less likely words,

but it comes with a risk.

In this case, the story starts out more originally,

about a young web artist from South Korea, but it quickly degenerates into nonsense.

Technically speaking, the API doesn't actually let you pick a temperature bigger than 2.

There's no mathematical reason for this, it's just an arbitrary constraint imposed

to keep their tool from being seen generating things that are too nonsensical.

So if you're curious, the way this animation is actually working is I'm taking the

20 most probable next tokens that GPT-3 generates,

which seems to be the maximum they'll give me,

and then I tweak the probabilities based on an exponent of 1/5.

As another bit of jargon, in the same way that you might call the components of

the output of this function probabilities, people often refer to the inputs as logits,

or some people say logits, some people say logits, I'm gonna say logits.

So for instance, when you feed in some text, you have all these word embeddings

flow through the network, and you do this final multiplication with the

unembedding matrix, machine learning people would refer to the components in that raw,

unnormalized output as the logits for the next word prediction.

A lot of the goal with this chapter was to lay the foundations for

understanding the attention mechanism, Karate Kid wax-on-wax-off style.

You see, if you have a strong intuition for word embeddings, for softmax,

for how dot products measure similarity, and also the underlying premise that

most of the calculations have to look like matrix multiplication with matrices

full of tunable parameters, then understanding the attention mechanism,

this cornerstone piece in the whole modern boom in AI, should be relatively smooth.

For that, come join me in the next chapter.

As I'm publishing this, a draft of that next chapter

is available for review by Patreon supporters.

A final version should be up in public in a week or two,

it usually depends on how much I end up changing based on that review.

In the meantime, if you want to dive into attention,

and if you want to help the channel out a little bit, it's there waiting.