import altair as alt 
import gradio as gr 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def disease_prediction(symptoms):
    prompt = f"Based on the following symptoms, provide possible medical conditions and general medication suggestions. Always emphasize the importance of consulting a doctor for proper diagnosis.\n\nSymptoms: {symptoms}\n\nPossible conditions and recommendations:\n\n**IMPORTANT: This is for informational purposes only. Please consult a healthcare professional for proper diagnosis and treatment.**\n\nAnalysis:"
    return generate_response(prompt, max_length=1200)

def treatment_plan(condition, age, gender, medical_history):
    prompt = f"Generate personalized treatment suggestions for the following patient information. Include home remedies and general medication guidelines.\n\nMedical Condition: {condition}\nAge: {age}\nGender: {gender}\nMedical History: {medical_history}\n\nPersonalized treatment plan including home remedies and medication guidelines:\n\n**IMPORTANT: This is for informational purposes only. Please consult a healthcare professional for proper treatment.**\n\nTreatment Plan:"
    return generate_response(prompt, max_length=1200)

def generate_health_insights(summary_string):
    """
    Generates health insights based on a summary string using the AI model.

    Args:
        summary_string: A string containing the summary of health data.

    Returns:
        A string containing the AI-generated health insights.
    """
    prompt = f"Based on the following health data summary, provide a brief overview and potential insights or areas for attention. Keep the response concise.\n\nHealth Data Summary:\n{summary_string}\n\nAI Insights:"
    return generate_response(prompt, max_length=800) # Adjust max_length as needed for conciseness


# Create Altair charts
heart_rate_chart = alt.Chart(df_health_trends).mark_line().encode(
    x=alt.X('Date', title='Date'),
    y=alt.Y('Heart Rate (7-Day Avg)', title='Heart Rate (bpm)'),
    tooltip=['Date', 'Heart Rate (7-Day Avg)']
).properties(
    title='Heart Rate Trend'
)

# Create a long-format DataFrame for blood pressure data
df_blood_pressure_long = df_health_trends.melt(
    id_vars=['Date'],
    value_vars=['Systolic (7-Day Avg)', 'Diastolic (7-Day Avg)'],
    var_name='BP Type',
    value_name='Value'
)

blood_pressure_chart = alt.Chart(df_blood_pressure_long).mark_line().encode(
    x=alt.X('Date', title='Date'),
    y=alt.Y('Value', title='Blood Pressure (mmHg)'),
    color='BP Type',
    tooltip=['Date', 'BP Type', 'Value']
).properties(
    title='Blood Pressure Trend'
)


blood_glucose_chart = alt.Chart(df_health_trends).mark_line().encode(
    x=alt.X('Date', title='Date'),
    y=alt.Y('Blood Glucose (7-Day Avg)', title='Blood Glucose (mg/dL)'),
    tooltip=['Date', 'Blood Glucose (7-Day Avg)']
).properties(
    title='Blood Glucose Trend'
)

symptom_frequency_chart = alt.Chart(symptom_frequency.reset_index().rename(columns={'index': 'Symptom', 'count': 'Frequency'})).mark_bar().encode(
    x=alt.X('Symptom', title='Symptom', sort='-y'),
    y=alt.Y('Frequency', title='Frequency'),
    tooltip=['Symptom', 'Frequency']
).properties(
    title='Symptom Frequency'
)


with gr.Blocks() as app:
    gr.Markdown("# Medical AI Assistant")
    gr.Markdown("**Disclaimer: This is for informational purposes only. Always consult healthcare professionals for medical advice.**")

    with gr.Tabs():
        with gr.TabItem("Welcome"):
            gr.Markdown("""
            ## Welcome to the Medical AI Assistant!

            This application provides informational assistance regarding potential medical conditions based on symptoms and offers personalized treatment suggestions.

            **Please remember:**
            * This tool is for informational purposes only and should not be considered a substitute for professional medical advice.
            * Always consult a qualified healthcare professional for any health concerns or before making any decisions related to your health or treatment.

            Navigate through the tabs above to explore the different functionalities:
            * **Disease Prediction:** Enter your symptoms to get a list of possible medical conditions and general recommendations.
            * **Treatment Plans:** Provide information about a medical condition and patient details to receive personalized treatment suggestions.
            """)
        with gr.TabItem("Disease Prediction"):
            with gr.Row():
                with gr.Column():
                    symptoms_input = gr.Textbox(
                        label="Enter Symptoms",
                        placeholder="e.g., fever, headache, cough, fatigue...",
                        lines=4
                    )
                    predict_btn = gr.Button("Analyze Symptoms")

                with gr.Column():
                    prediction_output = gr.Textbox(label="Possible Conditions & Recommendations", lines=20)

            predict_btn.click(disease_prediction, inputs=symptoms_input, outputs=prediction_output)

        with gr.TabItem("Treatment Plans"):
            with gr.Row():
                with gr.Column():
                    condition_input = gr.Textbox(
                        label="Medical Condition",
                        placeholder="e.g., diabetes, hypertension, migraine...",
                        lines=2
                    )
                    age_input = gr.Number(label="Age", value=30)
                    gender_input = gr.Dropdown(
                        choices=["Male", "Female", "Other"],
                        label="Gender",
                        value="Male"
                    )
                    history_input = gr.Textbox(
                        label="Medical History",
                        placeholder="Previous conditions, allergies, medications or None",
                        lines=3
                    )
                    plan_btn = gr.Button("Generate Treatment Plan")

                with gr.Column():
                    plan_output = gr.Textbox(label="Personalized Treatment Plan", lines=20)

            plan_btn.click(treatment_plan, inputs=[condition_input, age_input, gender_input, history_input], outputs=plan_output)

        with gr.TabItem("Health Analytics Dashboard") as dashboard_tab:
            gr.Markdown("## Health Analytics Overview")
            gr.Markdown("### Health Trends (Last 90 Days)")
            gr.Plot(heart_rate_chart)
            gr.Plot(blood_pressure_chart)
            gr.Plot(blood_glucose_chart)
            gr.Markdown("### Symptom Frequency (Last 90 Days)")
            gr.Plot(symptom_frequency_chart)
            gr.Markdown("### Health Metrics Summary")
            gr.Dataframe(health_metrics_summary)
            gr.Markdown("### AI-Generated Health Insights")
            insights_output = gr.Textbox(label="Insights", lines=10, interactive=False)

            # Generate and display insights when the tab is selected
            dashboard_tab.select(lambda: generate_health_insights(summary_string), inputs=None, outputs=insights_output)


app.launch(share=True)