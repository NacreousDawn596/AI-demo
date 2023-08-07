import random
import numpy as np
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras import backend as K
import keras
from os.path import exists

pickup_lines = [
    "Are you a campfire? Because you're hot and I want s'more.",
    "If you were a vegetable, you'd be a cute-cumber.",
    "Are you a Wi-Fi signal? Because I'm feeling a connection.",
    "Is your name Google? Because you have everything I've been searching for.",
    "Do you have a map? I just got lost in your eyes.",
    "Are you made of copper and tellurium? Because you're Cu-Te, and I'm rizz-struck!",
    "Is your name Netflix? Because I could watch you all day.",
    "Are you a parking ticket? Because you've got \"fine\" written all over you.",
    "If you were a fruit, you'd be a fineapple.",
    "Is your name Chapstick? Because you're da balm.",
    "Do you have a name, or can I call you mine?",
    "Are you a camera? Because every time I look at you, I smile.",
    "Are you an interior decorator? Because when I saw you, the room became beautiful.",
    "Can I follow you home? Cause my parents always told me to follow my dreams.",
    "Is your name Google Maps? Because you've got everything I'm searching for.",
    "Do you have a Band-Aid? Because I just scraped my knee falling for you.",
    "Is your name Wi-Fi? Because I'm really feeling a connection.",
    "Are you a campfire? Because you're hot and I want s'more.",
    "Can I take a picture of you to prove to my friends that angels exist?",
    "Do you believe in fate? Because I think we were rizz-destined to meet.",
    "If looks could kill, you'd be a weapon of mass destruction.",
    "Is your name Ariel? Because we mermaid for each other.",
    "Are you French? Because Eiffel for you.",
    "Can I follow you? Because my mom told me to follow my dreams.",
    "Do you have a sunburn, or are you always this hot?",
    "Is your name Pikachu? Because you are shockingly beautiful.",
    "Can you take a picture with me? I want to prove to my friends that angels are real.",
    "Is your name Cinderella? Because your beauty has me spellbound.",
    "Are you a loan? Because you have my interest.",
    "Do you believe in love at first sight, or should I walk by again?",
    "Are you a campfire? Because you're hot and I want s'more.",
    # "Can I borrow a pen? I want to write down the moment I met you.",
    "Is your name Honey? Because you're sweeter than sugar.",
    "Do you have a name, or can I call you mine?",
    "Are you a time traveler? Because I can see you in my future.",
    "Are you a campfire? Because you're hot and I want s'more.",
    "Are you a Wi-Fi signal? Because I'm really feeling a connection.",
    "Can you lend me a kiss? I promise I'll give it back.",
    "Is your name Chapstick? Because you're da balm.",
    "Do you have a map? I just got lost in your eyes.",
    "Is your name Google? Because you have everything I've been searching for.",
    "If you were a vegetable, you'd be a cute-cumber.",
    "Are you made of copper and tellurium? Because you're Cu-Te, and I'm rizz-struck!",
    "Is your name Netflix? Because I could watch you all day.",
    "Are you a parking ticket? Because you've got \"fine\" written all over you.",
    "If you were a fruit, you'd be a fineapple.",
    "Are you a camera? Because every time I look at you, I smile.",
    "Are you an interior decorator? Because when I saw you, the room became beautiful.",
    "Can I follow you home? Cause my parents always told me to follow my dreams.",
    "Is your name Google Maps? Because you've got everything I'm searching for.",
    "Do you have a Band-Aid? Because I just scraped my knee falling for you.",
    "Is your name Wi-Fi? Because I'm really feeling a connection.",
    "Are you a campfire? Because you're hot and I want s'more.",
    "Can I take a picture of you to prove to my friends that angels exist?",
    "Do you believe in fate? Because I think we were rizz-destined to meet.",
    "If looks could kill, you'd be a weapon of mass destruction.",
    "Is your name Ariel? Because we mermaid for each other.",
    "Are you French? Because Eiffel for you.",
    "Can I follow you? Because my mom told me to follow my dreams.",
    "Do you have a sunburn, or are you always this hot?",
    "Is your name Pikachu? Because you are shockingly beautiful.",
    "Can you take a picture with me? I want to prove to my friends that angels are real.",
    "Is your name Cinderella? Because your beauty has me spellbound.",
    "Are you a loan? Because you have my interest.",
    "Do you believe in love at first sight, or should I walk by again?",
    "Are you a campfire? Because you're hot and I want s'more.",
    "Can I borrow a pen? I want to write down the moment I met you.",
    "Is your name Honey? Because you're sweeter than sugar.",
    "Do you have a name, or can I call you mine?",
    "Are you a time traveler? Because I can see you in my future.",
    "Are you a campfire? Because you're hot and I want s'more.",
    "Are you a Wi-Fi signal? Because I'm really feeling a connection.",
    "Can you lend me a kiss? I promise I'll give it back.",
    "Is your name Chapstick? Because you're da balm.",
    "Do you have a map? I just got lost in your eyes.",
    "Is your name Google? Because you have everything I've been searching for.",
    "If you were a vegetable, you'd be a cute-cumber.",
    "Are you made of copper and tellurium? Because you're Cu-Te, and I'm rizz-struck!",
    "Is your name Netflix? Because I could watch you all day.",
    "Are you a parking ticket? Because you've got \"fine\" written all over you.",
    "If you were a fruit, you'd be a fineapple.",
    "Are you a camera? Because every time I look at you, I smile.",
    "Are you an interior decorator? Because when I saw you, the room became beautiful.",
    "Can I follow you home? Cause my parents always told me to follow my dreams.",
    "Is your name Google Maps? Because you've got everything I'm searching for.",
    "Do you have a Band-Aid? Because I just scraped my knee falling for you.",
    "Is your name Wi-Fi? Because I'm really feeling a connection.",
    "Are you a campfire? Because you're hot and I want s'more.",
    "Can I take a picture of you to prove to my friends that angels exist?",
    "Do you believe in fate? Because I think we were rizz-destined to meet.",
    "If looks could kill, you'd be a weapon of mass destruction.",
    "Is your name Ariel? Because we mermaid for each other.",
    "Are you French? Because Eiffel for you.",
    "Can I follow you? Because my mom told me to follow my dreams.",
    "Do you have a sunburn, or are you always this hot?",
    "Is your name Pikachu? Because you are shockingly beautiful.",
    "Can you take a picture with me? I want to prove to my friends that angels are real.",
    "Is your name Cinderella? Because your beauty has me spellbound.",
    "Are you a loan? Because you have my interest.",
    "Do you believe in love at first sight, or should I walk by again?"
]

vocab = sorted(set(" ".join(pickup_lines)))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = np.array(vocab)

if not exists("pickup_model"):
    pickup_lines_int = [[char_to_idx[char] for char in line] for line in pickup_lines]
    input_sequences = []
    target_sequences = []
    sequence_length = 40

    for line_int in pickup_lines_int:
        for i in range(len(line_int) - sequence_length):
            input_sequences.append(line_int[i : i + sequence_length])
            target_sequences.append(line_int[i + 1 : i + sequence_length + 1])

    input_sequences = np.array(input_sequences)
    target_sequences = np.array(target_sequences)

    model = Sequential([
        Embedding(len(vocab), 128, batch_input_shape=[1, None]),
        LSTM(256, return_sequences=True, recurrent_initializer="glorot_uniform", stateful=True, batch_input_shape=[1, None, 128]),
        Dense(len(vocab), activation="softmax")
    ])

    model.compile(optimizer=RMSprop(learning_rate=0.1), loss="sparse_categorical_crossentropy")

    model.fit(input_sequences, target_sequences, epochs=100, batch_size=1)
    model.save("pickup_model")
else:
    model = keras.models.load_model("pickup_model")

beam_width = 5
temperature = 0.1
num_generate = random.randint(50, 100)

class BeamSearchNode:
    def __init__(self, sequence, score):
        self.sequence = sequence
        self.score = score

def beam_search(model, start_sequence, beam_width, max_length, temperature):
    queue = [BeamSearchNode(start_sequence, 0)]

    for _ in range(max_length):
        new_queue = []

        for node in queue:
            input_sequence = np.array([node.sequence])
            predictions = model.predict(input_sequence)
            predicted_probs = np.log(predictions[0, -1, :]) / temperature
            predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

            top_indices = np.argsort(predicted_probs)[-beam_width:]
            for idx in top_indices:
                new_sequence = node.sequence + [idx]
                new_score = node.score - np.log(predicted_probs[idx])
                new_node = BeamSearchNode(new_sequence, new_score)
                new_queue.append(new_node)

        new_queue.sort(key=lambda n: n.score)
        queue = new_queue[:beam_width]

    return queue[0].sequence

start_sentence = ' '.join(random.choice(pickup_lines).split()[0:7])
start_sequence = [char_to_idx[char] for char in start_sentence]

generated_sequence = beam_search(model, start_sequence, beam_width, num_generate, temperature)
generated_sentence = ' '.join([idx_to_char[idx] for idx in generated_sequence])

print(generated_sentence)