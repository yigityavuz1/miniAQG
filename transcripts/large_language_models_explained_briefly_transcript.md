# Transcript for https://www.youtube.com/watch?v=LPZh9BOjkQs


Kind: captions

Language: en

Imagine you happen across a short movie script that

describes a scene between a person and their AI assistant.

The script has what the person asks the AI, but the AI's response has been torn off.

Suppose you also have this powerful magical machine that can take

any text and provide a sensible prediction of what word comes next.

You could then finish the script by feeding in what you have to the machine,

seeing what it would predict to start the AI's answer,

and then repeating this over and over with a growing script completing the dialogue.

When you interact with a chatbot, this is exactly what's happening.

A large language model is a sophisticated mathematical function

that predicts what word comes next for any piece of text.

Instead of predicting one word with certainty, though,

what it does is assign a probability to all possible next words.

To build a chatbot, you lay out some text that describes an interaction between a user

and a hypothetical AI assistant, add on whatever the user types in as the first part of

the interaction, and then have the model repeatedly predict the next word that such a

hypothetical AI assistant would say in response, and that's what's presented to the user.

In doing this, the output tends to look a lot more natural if

you allow it to select less likely words along the way at random.

So what this means is even though the model itself is deterministic,

a given prompt typically gives a different answer each time it's run.

Models learn how to make these predictions by processing an enormous amount of text,

typically pulled from the internet.

For a standard human to read the amount of text that was used to train GPT-3,

for example, if they read non-stop 24-7, it would take over 2600 years.

Larger models since then train on much, much more.

You can think of training a little bit like tuning the dials on a big machine.

The way that a language model behaves is entirely determined by these

many different continuous values, usually called parameters or weights.

Changing those parameters will change the probabilities

that the model gives for the next word on a given input.

What puts the large in large language model is how

they can have hundreds of billions of these parameters.

No human ever deliberately sets those parameters.

Instead, they begin at random, meaning the model just outputs gibberish,

but they're repeatedly refined based on many example pieces of text.

One of these training examples could be just a handful of words,

or it could be thousands, but in either case, the way this works is to

pass in all but the last word from that example into the model and

compare the prediction that it makes with the true last word from the example.

An algorithm called backpropagation is used to tweak all of the parameters

in such a way that it makes the model a little more likely to choose

the true last word and a little less likely to choose all the others.

When you do this for many, many trillions of examples,

not only does the model start to give more accurate predictions on the training data,

but it also starts to make more reasonable predictions on text that it's never

seen before.

Given the huge number of parameters and the enormous amount of training data,

the scale of computation involved in training a large language model is mind-boggling.

To illustrate, imagine that you could perform one

billion additions and multiplications every single second.

How long do you think it would take for you to do all of the

operations involved in training the largest language models?

Do you think it would take a year?

Maybe something like 10,000 years?

The answer is actually much more than that.

It's well over 100 million years.

This is only part of the story, though.

This whole process is called pre-training.

The goal of auto-completing a random passage of text from the

internet is very different from the goal of being a good AI assistant.

To address this, chatbots undergo another type of training,

just as important, called reinforcement learning with human feedback.

Workers flag unhelpful or problematic predictions,

and their corrections further change the model's parameters,

making them more likely to give predictions that users prefer.

Looking back at the pre-training, though, this staggering amount of

computation is only made possible by using special computer chips that

are optimized for running many operations in parallel, known as GPUs.

However, not all language models can be easily parallelized.

Prior to 2017, most language models would process text one word at a time,

but then a team of researchers at Google introduced a new model known as the transformer.

Transformers don't read text from the start to the finish,

they soak it all in at once, in parallel.

The very first step inside a transformer, and most other language models for that matter,

is to associate each word with a long list of numbers.

The reason for this is that the training process only works with continuous values,

so you have to somehow encode language using numbers,

and each of these lists of numbers may somehow encode the meaning of the

corresponding word.

What makes transformers unique is their reliance

on a special operation known as attention.

This operation gives all of these lists of numbers a chance to talk to one another

and refine the meanings they encode based on the context around, all done in parallel.

For example, the numbers encoding the word bank might be changed based on the

context surrounding it to somehow encode the more specific notion of a riverbank.

Transformers typically also include a second type of operation known

as a feed-forward neural network, and this gives the model extra

capacity to store more patterns about language learned during training.

All of this data repeatedly flows through many different iterations of

these two fundamental operations, and as it does so,

the hope is that each list of numbers is enriched to encode whatever

information might be needed to make an accurate prediction of what word

follows in the passage.

At the end, one final function is performed on the last vector in this sequence,

which now has had a chance to be influenced by all the other context from the input text,

as well as everything the model learned during training,

to produce a prediction of the next word.

Again, the model's prediction looks like a probability for every possible next word.

Although researchers design the framework for how each of these steps work,

it's important to understand that the specific behavior is an emergent phenomenon

based on how those hundreds of billions of parameters are tuned during training.

This makes it incredibly challenging to determine

why the model makes the exact predictions that it does.

What you can see is that when you use large language model predictions to autocomplete

a prompt, the words that it generates are uncannily fluent, fascinating, and even useful.

If you're a new viewer and you're curious about more details on how

transformers and attention work, boy do I have some material for you.

One option is to jump into a series I made about deep learning,

where we visualize and motivate the details of attention and all the other steps

in a transformer.

Also, on my second channel I just posted a talk I gave a couple

months ago about this topic for the company TNG in Munich.

Sometimes I actually prefer the content I make as a casual talk rather than a produced

video, but I leave it up to you which one of these feels like the better follow-on.