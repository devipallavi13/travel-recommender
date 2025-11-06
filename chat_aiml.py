# chat_aiml.py
import aiml
import os
from recommender_ml import recommend_by_preferences

kernel = aiml.Kernel()
kernel.learn("aiml/travel.aiml")
kernel.respond("HI")  # optional

print("AIML Travel Bot: Say 'HI' to start. Type 'exit' to quit.")

while True:
    user = input("YOU: ")
    if user.strip().lower() in ('exit','quit'):
        break
    response = kernel.respond(user)
    if response.strip() == "GET_RECOMMS":
        trip_type = kernel.getPredicate('trip_type') or ""
        budget = kernel.getPredicate('budget') or None
        month = kernel.getPredicate('month') or None
        prefs = [t.strip() for t in trip_type.split() if t.strip()]
        recs = recommend_by_preferences(preferred_types=prefs, budget=budget, month=month, top_k=5)
        if recs.empty:
            print("BOT: I couldn't find matches. Try relaxing the budget or changing trip type.")
        else:
            print("BOT: Here are recommendations:")
            for i, r in recs.iterrows():
                print(f" {i+1}. {r['Destination']} ({r['Country']}) - {r['Type']} - ${r['Average_Cost']}/day - Rating: {r['Rating']:.1f}")
    else:
        print("BOT:", response)
