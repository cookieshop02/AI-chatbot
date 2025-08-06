from flask import Flask, request, jsonify, render_template, session
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import random
import json
import uuid
import re


# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

app = Flask(__name__)
app.secret_key = "chatbot_secret_key"  # Required for session management

sia = SentimentIntensityAnalyzer()

# Track conversation state
conversation_states = {}

# DASS-21 Questionnaire setup
dass21_questions = [
    # Depression items
    "I couldn't seem to experience any positive feeling at all.",
    "I found it difficult to work up the initiative to do things.",
    "I felt that I had nothing to look forward to.",
    "I felt down-hearted and blue.",
    "I was unable to become enthusiastic about anything.",
    "I felt I wasn't worth much as a person.",
    "I felt that life was meaningless.",
    
    # Anxiety items
    "I was aware of dryness of my mouth.",
    "I experienced breathing difficulty (e.g., excessively rapid breathing, breathlessness in the absence of physical exertion).",
    "I experienced trembling (e.g., in the hands).",
    "I was worried about situations in which I might panic and make a fool of myself.",
    "I felt I was close to panic.",
    "I was aware of the action of my heart in the absence of physical exertion (e.g., sense of heart rate increase, heart missing a beat).",
    "I felt scared without any good reason.",
    
    # Stress items
    "I found it hard to wind down.",
    "I tended to over-react to situations.",
    "I felt that I was using a lot of nervous energy.",
    "I found myself getting agitated.",
    "I found it difficult to relax.",
    "I was intolerant of anything that kept me from getting on with what I was doing.",
    "I felt that I was rather touchy."
]

dass21_responses = [
    "Did not apply to me at all",
    "Applied to me to some degree, or some of the time",
    "Applied to me to a considerable degree, or a good part of time",
    "Applied to me very much, or most of the time"
]

# Candidate responses for anxiety intervention help suggestions
anxiety_responses = {
    "It seems you're feeling anxious. Have you tried deep breathing exercises?": 0.0,
    "Sometimes physical activity can help. A short walk or some light exercise might improve your mood.": 0.0,
    "Mindfulness meditation could be beneficial. Try focusing on your breath.": 0.0,
    "It might help to talk about what you're feeling. Consider reaching out to a friend or a professional for support.": 0.0,
    "Writing down your thoughts in a journal can be therapeutic and may help reduce anxiety.": 0.0
}

# Follow-up responses for when someone says "no" to anxiety suggestions
anxiety_followups = {
    "I understand. Deep breathing isn't for everyone. Would you like to try another approach? Perhaps talking about what's making you anxious might help.": 0.0,
    "That's okay. Sometimes it helps to identify what's triggering your anxiety. Can you share what's on your mind?": 0.0,
    "No problem. There are many ways to manage anxiety. Have you found anything that helps you feel calmer in the past?": 0.0,
    "I understand. Would you prefer to try mindfulness meditation instead? It can be helpful for managing anxiety.": 0.0,
    "That's alright. How about trying to focus on something positive? Is there something you're looking forward to?": 0.0
}

# Adding positive responses for happy/good mood expressions
positive_responses = {
    "That's wonderful to hear! It's great that you're feeling good today.": 0.0,
    "I'm so happy to hear that! What's contributing to your positive mood?": 0.0,
    "That's excellent! Positive feelings are worth celebrating. Keep it up!": 0.0,
    "Great to hear you're in a good mood! Is there anything specific that made your day better?": 0.0,
    "Fantastic! Happiness is contagious - thanks for sharing your positive energy!": 0.0
}

# Adding responses for sad mood expressions
sad_responses = {
    "I'm sorry to hear you're feeling sad. Would you like to talk about what's bothering you?": 0.0,
    "It's okay to feel sad sometimes. Is there anything I can do to support you?": 0.0,
    "I'm here for you. Sometimes sharing what's making you sad can help lighten the burden.": 0.0,
    "I understand. Sadness is a natural emotion. Is there something specific that's causing you to feel this way?": 0.0,
    "Thank you for sharing how you're feeling. Would talking about it help you feel better?": 0.0
}

# Adding responses for stress expressions
stress_responses = {
    "I can see you're feeling stressed. Taking a few moments to breathe deeply might help.": 0.0,
    "Stress can be challenging. Would you like to talk about what's causing it?": 0.0,
    "When you're feeling stressed, sometimes a short break can help. Could you step away for 5 minutes?": 0.0,
    "I understand stress can be difficult. Have you tried any relaxation techniques today?": 0.0,
    "Feeling stressed is common. Would it help to identify what's triggering your stress?": 0.0
}

# Follow-up responses for stress suggestions
stress_followups = {
    "I understand relaxation techniques don't work for everyone. Is there something specific causing your stress that you'd like to discuss?": 0.0,
    "That's okay. Sometimes identifying the source of stress can help manage it. What's been on your mind lately?": 0.0,
    "No problem. Everyone manages stress differently. What has helped you feel less stressed in the past?": 0.0,
    "I understand. Would it help to talk about ways to address the specific situation that's causing your stress?": 0.0,
    "That's alright. Sometimes just acknowledging stress is the first step. Is there anything else you'd like to talk about?": 0.0
}

# General conversation responses for when no emotional state is detected
general_conversation_responses = {
    "What aspects of mental health are you most interested in discussing today?": 0.0,
    "Is there something specific about wellbeing or mental health you'd like to explore?": 0.0,
    "I'm here to chat about various topics related to mental wellness. What's on your mind?": 0.0,
    "I'd be happy to discuss coping strategies or mental health topics that interest you.": 0.0,
    "Everyone's mental health journey is unique. Is there something particular you're curious about?": 0.0
}

# Common questions and their responses
faq_responses = {
    "what can you do": "I can chat with you about how you're feeling, offer support for emotions like anxiety or stress, and even administer the DASS-21 questionnaire to help assess depression, anxiety, and stress levels. Just type 'DASS-21' if you'd like to take the assessment.",
    "who made you": "I'm a mental health chatbot designed to provide supportive conversations and basic assessments. I use reinforcement learning to improve my responses over time.",
    "how does this work": "I analyze your messages for emotional content and try to provide supportive responses. I can recognize feelings like anxiety, stress, sadness, and happiness, and offer appropriate support. I also learn from our interactions to improve over time.",
    "what is dass21": "The DASS-21 is a 21-item questionnaire that measures symptoms of depression, anxiety, and stress. It's a shorter version of the DASS-42. It's not a diagnostic tool, but it can help identify symptoms that might warrant professional attention. Type 'DASS-21' if you'd like to take it.",
    "help": "I can help by chatting about your feelings, offering supportive responses, or administering the DASS-21 assessment. Just tell me how you're feeling or what's on your mind. Type 'DASS-21' if you'd like to take the assessment."
}

# Responses after completing DASS-21
dass21_feedback_responses = {
    "high_depression": "Based on your responses, you may be experiencing symptoms of depression. Consider speaking with a mental health professional for further evaluation and support.",
    "moderate_depression": "Your responses indicate some symptoms of depression. Taking time for self-care and seeking support could be beneficial.",
    "mild_depression": "You're showing some mild symptoms of depression. Self-care strategies might help improve your mood.",
    "normal_depression": "Your depression score is within the normal range.",
    
    "high_anxiety": "Your responses suggest significant anxiety symptoms. A mental health professional could provide strategies to help manage these feelings.",
    "moderate_anxiety": "You're showing moderate anxiety symptoms. Learning anxiety management techniques could be helpful.",
    "mild_anxiety": "Your responses indicate mild anxiety. Simple relaxation techniques might be beneficial.",
    "normal_anxiety": "Your anxiety score is within the normal range.",
    
    "high_stress": "Your responses indicate high stress levels. It's important to find healthy ways to manage stress.",
    "moderate_stress": "You're experiencing moderate stress levels. Stress management techniques could be helpful.",
    "mild_stress": "Your responses show mild stress. Simple self-care strategies might help reduce this.",
    "normal_stress": "Your stress score is within the normal range."
}

def save_q_values(filepath="q_values.json"):
    """Save the response dictionaries (with Q-values) to a JSON file."""
    data = {
        "anxiety_responses": anxiety_responses,
        "anxiety_followups": anxiety_followups,
        "positive_responses": positive_responses,
        "sad_responses": sad_responses,
        "stress_responses": stress_responses,
        "stress_followups": stress_followups,
        "general_conversation_responses": general_conversation_responses
    }
    with open(filepath, "w") as f:
        json.dump(data, f)

def load_q_values(filepath="q_values.json"):
    """Load the response dictionaries (with Q-values) from a JSON file, if available."""
    global anxiety_responses, anxiety_followups, positive_responses, sad_responses, stress_responses, stress_followups, general_conversation_responses
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            anxiety_responses.update(data.get("anxiety_responses", {}))
            anxiety_followups.update(data.get("anxiety_followups", {}))
            positive_responses.update(data.get("positive_responses", {}))
            sad_responses.update(data.get("sad_responses", {}))
            stress_responses.update(data.get("stress_responses", {}))
            stress_followups.update(data.get("stress_followups", {}))
            general_conversation_responses.update(data.get("general_conversation_responses", {}))
    except FileNotFoundError:
        pass

# Load saved Q-values at startup, if any
load_q_values()

def select_response(candidate_dict):
    """
    Select a candidate response using an epsilon-greedy strategy.
    With probability epsilon, choose a random response (exploration);
    otherwise, choose the response with the highest Q-value (exploitation).
    """
    epsilon = 0.1  # 10% chance to explore randomly
    if random.random() < epsilon:
        return random.choice(list(candidate_dict.keys()))
    else:
        return max(candidate_dict, key=candidate_dict.get)

def update_q_value(candidate_dict, response, reward, learning_rate=0.1):
    """
    Update the Q-value for the selected response based on the reward.
    Q_new = Q_old + learning_rate * (reward - Q_old)
    """
    candidate_dict[response] = candidate_dict[response] + learning_rate * (reward - candidate_dict[response])
    save_q_values()  # Save updated Q-values to file

def contains_anxiety_keywords(user_input):
    """
    Check if the user input contains any anxiety-related keywords.
    """
    anxiety_keywords = ["anxiety", "anxious", "nervous", "panic", "worried", "fear", "tense", "worry", "afraid", 
                        "uneasy", "apprehensive", "frightened", "scared"]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in anxiety_keywords)

def contains_stress_keywords(user_input):
    """
    Check if the user input contains stress-related keywords.
    """
    stress_keywords = ["stress", "stressed", "stressful", "pressure", "overwhelm", "overwhelmed", 
                      "burnt out", "burnout", "tension", "exhausted", "overworked", "too much"]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in stress_keywords)

def contains_positive_keywords(user_input):
    """
    Check if the user input contains positive mood keywords.
    """
    positive_keywords = ["happy", "good mood", "great", "excellent", "wonderful", "joyful", "fantastic", 
                         "feeling good", "feeling better", "cheerful", "positive", "upbeat", "content"]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in positive_keywords)

def contains_sad_keywords(user_input):
    """
    Check if the user input contains sad mood keywords.
    """
    sad_keywords = ["sad", "unhappy", "depressed", "down", "blue", "miserable", "upset", 
                    "gloomy", "heartbroken", "disappointed", "sorrowful", "hurt"]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in sad_keywords)

def contains_negative_response(user_input):
    """
    Check if the user input contains negative responses like "no".
    """
    negative_responses = ["no", "nope", "don't want to", "not really", "not interested", "haven't"]
    user_input_lower = user_input.lower()
    return any(response in user_input_lower for response in negative_responses)

def contains_dass21_command(user_input):
    """
    Check if the user input contains a request to take the DASS-21 test.
    """
    dass21_keywords = ["dass", "dass21", "dass-21", "depression test", "anxiety test", "stress test", 
                       "mental health test", "assessment", "questionnaire", "test me"]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in dass21_keywords)

def check_for_faq(user_input):
    """
    Check if the user input matches any FAQ and return the appropriate response.
    """
    user_input_lower = user_input.lower()
    
    # Check for exact matches first
    for question, answer in faq_responses.items():
        if question in user_input_lower:
            return answer
    
    # Check for question types
    if re.search(r"what (can|do) you do", user_input_lower) or re.search(r"how (can|do) you help", user_input_lower):
        return faq_responses["what can you do"]
    
    if re.search(r"who (made|created|developed) you", user_input_lower) or "who are you" in user_input_lower:
        return faq_responses["who made you"]
    
    if re.search(r"how (does|do) (this|you|it) work", user_input_lower):
        return faq_responses["how does this work"]
    
    if re.search(r"what is (dass|dass21|dass-21)", user_input_lower) or "depression test" in user_input_lower:
        return faq_responses["what is dass21"]
    
    # No FAQ match found
    return None

def interpret_dass21_scores(depression_score, anxiety_score, stress_score):
    """
    Interpret DASS-21 scores based on standard severity ratings.
    Returns a tuple of (depression_level, anxiety_level, stress_level)
    """
    # Depression interpretation (multiply by 2 to match full DASS-42 scale)
    depression_score *= 2
    if depression_score >= 28:
        depression_level = "high_depression"
    elif depression_score >= 20:
        depression_level = "moderate_depression"
    elif depression_score >= 10:
        depression_level = "mild_depression"
    else:
        depression_level = "normal_depression"
    
    # Anxiety interpretation (multiply by 2 to match full DASS-42 scale)
    anxiety_score *= 2
    if anxiety_score >= 20:
        anxiety_level = "high_anxiety"
    elif anxiety_score >= 14:
        anxiety_level = "moderate_anxiety"
    elif anxiety_score >= 8:
        anxiety_level = "mild_anxiety"
    else:
        anxiety_level = "normal_anxiety"
    
    # Stress interpretation (multiply by 2 to match full DASS-42 scale)
    stress_score *= 2
    if stress_score >= 34:
        stress_level = "high_stress"
    elif stress_score >= 26:
        stress_level = "moderate_stress"
    elif stress_score >= 18:
        stress_level = "mild_stress"
    else:
        stress_level = "normal_stress"
    
    return depression_level, anxiety_level, stress_level

@app.route("/")
def home():
    # Generate a session ID if it doesn't exist
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    
    # Initialize conversation state for this session
    if session["session_id"] not in conversation_states:
        conversation_states[session["session_id"]] = {
            "last_question_type": None,
            "last_response": None,
            "in_dass21": False,
            "dass21_question_index": 0,
            "dass21_scores": {
                "depression": 0,
                "anxiety": 0,
                "stress": 0
            },
            "consecutive_default_responses": 0,
            "conversation_history": []
        }
    
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()
    
    # Get or create session ID
    session_id = session.get("session_id", str(uuid.uuid4()))
    if "session_id" not in session:
        session["session_id"] = session_id
    
    # Get or initialize conversation state
    if session_id not in conversation_states:
        conversation_states[session_id] = {
            "last_question_type": None,
            "last_response": None,
            "in_dass21": False,
            "dass21_question_index": 0,
            "dass21_scores": {
                "depression": 0,
                "anxiety": 0,
                "stress": 0
            },
            "consecutive_default_responses": 0,
            "conversation_history": []
        }
    
    state = conversation_states[session_id]
    
    # Record this message in conversation history
    state["conversation_history"].append({"role": "user", "content": user_input})
    
    # Handle DASS-21 questionnaire
    if state["in_dass21"]:
        # Try to process the answer as a number (0-3) or text matching the options
        answer = -1
        try:
            # Try to parse as number first
            answer = int(user_input)
            if answer < 0 or answer > 3:
                response = "Please enter a number between 0 and 3, where:\n0 = Did not apply to me at all\n1 = Applied to me to some degree\n2 = Applied to me considerably\n3 = Applied to me very much"
                state["conversation_history"].append({"role": "bot", "content": response})
                return jsonify({"response": response})
        except ValueError:
            # Try to match to response options
            user_input_lower = user_input.lower()
            for i, response in enumerate(dass21_responses):
                if user_input_lower in response.lower():
                    answer = i
                    break
            
            # If still not matched, handle simple cases
            if answer == -1:
                if user_input_lower in ["0", "not at all", "never", "none"]:
                    answer = 0
                elif user_input_lower in ["1", "somewhat", "sometimes", "some"]:
                    answer = 1
                elif user_input_lower in ["2", "considerable", "often", "good part"]:
                    answer = 2
                elif user_input_lower in ["3", "very much", "always", "most of the time"]:
                    answer = 3
        
        # If we couldn't parse the answer, ask again
        if answer == -1:
            response = "I didn't understand your response. Please enter a number between 0-3:\n0 = Did not apply to me at all\n1 = Applied to me to some degree\n2 = Applied to me considerably\n3 = Applied to me very much"
            state["conversation_history"].append({"role": "bot", "content": response})
            return jsonify({"response": response})
        
        # Record the score in the appropriate category
        question_index = state["dass21_question_index"]
        if question_index < 7:
            state["dass21_scores"]["depression"] += answer
        elif question_index < 14:
            state["dass21_scores"]["anxiety"] += answer
        else:
            state["dass21_scores"]["stress"] += answer
        
        # Move to the next question or finish the questionnaire
        state["dass21_question_index"] += 1
        if state["dass21_question_index"] < len(dass21_questions):
            # Still have more questions
            question_num = state["dass21_question_index"] + 1
            question = dass21_questions[state["dass21_question_index"]]
            response = f"Question {question_num}/{len(dass21_questions)}: {question}\n\nPlease rate on a scale of 0-3 how much this applied to you in the past week:\n0 = Did not apply to me at all\n1 = Applied to me to some degree\n2 = Applied to me considerably\n3 = Applied to me very much"
            state["conversation_history"].append({"role": "bot", "content": response})
            return jsonify({"response": response})
        else:
            # Questionnaire completed
            state["in_dass21"] = False
            depression_score = state["dass21_scores"]["depression"]
            anxiety_score = state["dass21_scores"]["anxiety"]
            stress_score = state["dass21_scores"]["stress"]
            
            # Interpret scores
            depression_level, anxiety_level, stress_level = interpret_dass21_scores(
                depression_score, anxiety_score, stress_score
            )
            
            # Reset for next time
            state["dass21_question_index"] = 0
            state["dass21_scores"] = {"depression": 0, "anxiety": 0, "stress": 0}
            
            # Provide feedback
            feedback = f"Thank you for completing the DASS-21 questionnaire. Here are your results:\n\n"
            feedback += f"Depression score: {depression_score*2}/42\n{dass21_feedback_responses[depression_level]}\n\n"
            feedback += f"Anxiety score: {anxiety_score*2}/42\n{dass21_feedback_responses[anxiety_level]}\n\n"
            feedback += f"Stress score: {stress_score*2}/42\n{dass21_feedback_responses[stress_level]}\n\n"
            feedback += "Remember, this is not a clinical diagnosis. If you're concerned about your mental health, please speak with a qualified mental health professional."
            
            state["conversation_history"].append({"role": "bot", "content": feedback})
            return jsonify({"response": feedback})
    
    # Check if user wants to start DASS-21
    if contains_dass21_command(user_input):
        state["in_dass21"] = True
        state["dass21_question_index"] = 0
        state["dass21_scores"] = {"depression": 0, "anxiety": 0, "stress": 0}
        state["consecutive_default_responses"] = 0
        
        intro = "I'll help you take the DASS-21 questionnaire, which measures depression, anxiety, and stress symptoms. It has 21 questions that refer to how you've been feeling during the past week.\n\n"
        intro += "For each statement, please rate on a scale of 0-3 how much it applied to you:\n"
        intro += "0 = Did not apply to me at all\n"
        intro += "1 = Applied to me to some degree, or some of the time\n"
        intro += "2 = Applied to me to a considerable degree, or a good part of time\n"
        intro += "3 = Applied to me very much, or most of the time\n\n"
        intro += f"Question 1/{len(dass21_questions)}: {dass21_questions[0]}"
        
        state["conversation_history"].append({"role": "bot", "content": intro})
        return jsonify({"response": intro})
    
    # Check if the input matches any FAQ
    faq_response = check_for_faq(user_input)
    if faq_response:
        state["consecutive_default_responses"] = 0
        state["last_question_type"] = "faq"
        state["conversation_history"].append({"role": "bot", "content": faq_response})
        return jsonify({"response": faq_response})
    
    # Analyze sentiment for reward calculation
    sentiment_scores = sia.polarity_scores(user_input)
    compound = sentiment_scores['compound']
    
    # Define reward based on sentiment
    if compound >= 0.05:
        reward = 1
    elif compound <= -0.05:
        reward = -1
    else:
        reward = 0
    
    # Handle follow-up responses based on previous question type
    if state["last_question_type"] == "anxiety" and contains_negative_response(user_input):
        selected_response = select_response(anxiety_followups)
        update_q_value(anxiety_followups, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "anxiety_followup"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    
    # Handle stress follow-ups
    if state["last_question_type"] == "stress" and contains_negative_response(user_input):
        selected_response = select_response(stress_followups)
        update_q_value(stress_followups, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "stress_followup"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    
    # Basic greeting response
    user_input_lower = user_input.lower()
    greetings = ["hi", "hello", "hey"]
    if any(greeting == user_input_lower for greeting in greetings):
        response = "Hi, how are you? I'm here to help. How are you feeling today? If you'd like to take the DASS-21 questionnaire to assess depression, anxiety, and stress, just type 'DASS-21'."
        state["last_question_type"] = "greeting"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": response})
        return jsonify({"response": response})
    
    # Handle emotion-specific responses
    if contains_positive_keywords(user_input):
        selected_response = select_response(positive_responses)
        update_q_value(positive_responses, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "positive"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    
    elif contains_sad_keywords(user_input):
        selected_response = select_response(sad_responses)
        update_q_value(sad_responses, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "sad"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    
    elif contains_anxiety_keywords(user_input):
        selected_response = select_response(anxiety_responses)
        update_q_value(anxiety_responses, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "anxiety"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    
    elif contains_stress_keywords(user_input):
        selected_response = select_response(stress_responses)
        update_q_value(stress_responses, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "stress"
        state["consecutive_default_responses"] = 0
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    
    # If we've already given the default response multiple times, try a different approach
    if state["consecutive_default_responses"] >= 2:
        selected_response = select_response(general_conversation_responses)
        update_q_value(general_conversation_responses, selected_response, reward)
        state["last_response"] = selected_response
        state["last_question_type"] = "general"
        state["consecutive_default_responses"] += 1
        state["conversation_history"].append({"role": "bot", "content": selected_response})
        return jsonify({"response": selected_response})
    else:
        # Default response if no specific emotion is detected
        response = "I'm here to support you. How are you feeling today? Whether you're having a great day or facing some challenges, I'm here to chat. You can also take the DASS-21 questionnaire by typing 'DASS-21'."
        state["last_question_type"] = "default"
        state["consecutive_default_responses"] += 1
        state["conversation_history"].append({"role": "bot", "content": response})
        return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)