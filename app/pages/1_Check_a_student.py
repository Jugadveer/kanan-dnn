"""Enter one student's first-term grade and details, see the predicted group."""

import streamlit as st

from earlywarning import settings
from shared import GROUP_COLOURS, grade_range, group_lamp, load_predictor, meter, risk_colour, start_page

SUPPORT = 0  # index of "Needs support" in settings.CATEGORIES
# What to do about each group; only Needs support asks for action.
ADVICE = [
    "Worth acting on now: extra lessons, a mentor, a check on attendance.",
    "",
    "",
]
# How sure the wording sounds. The probabilities were calibrated on validation
# students, so "70 %" means right about 7 times in 10 and can be read at face value.
SURE, PROBABLE = 0.70, 0.50


def ask(field: dict, label: str | None = None):
    """One input, limited to values seen in training. yes/no -> toggle,
    ordered scale -> select slider, other choices -> dropdown, counts -> number box."""
    question, default, key = label or field["question"], field["default"], field["column"]
    if "choices" not in field:
        return st.number_input(question, field["min"], field["max"], default, step=1, key=key)

    words = dict((value, label) for value, label in field["choices"])
    values = list(words)
    if set(values) == {"no", "yes"}:
        return "yes" if st.toggle(question, value=default == "yes", key=key) else "no"
    if all(isinstance(value, int) for value in values):
        return st.select_slider(question, values, value=default, format_func=words.get, key=key)
    return st.selectbox(question, values, index=values.index(default), format_func=words.get, key=key)


def first_term_grade(field: dict) -> int:
    """The strongest input comes first, as a plain number box."""
    grade = st.number_input("First-term grade in Portuguese (G1, 0–20)", field["min"], field["max"],
                            field["default"], step=1, key="G1")
    group = settings.CATEGORIES[sum(grade > ceiling for ceiling in settings.GRADE_CEILINGS[:-1])]
    st.caption(f"On its own this grade sits in **{group}**. About one student in four ends the "
               f"year in a different group, so the network gives chances, not just the group.")
    return grade


def student_inputs(form: list[dict]) -> dict:
    """G1 first, then the other details in one tab per group from the brief.
    Every tab is drawn on each run, so all answers are known even when hidden."""
    fields = {field["column"]: field for field in form}
    student = {"G1": first_term_grade(fields["G1"])} if "G1" in fields else {}
    if len(student) == len(form):
        st.info("Feature selection kept only the first-term grade: none of the 29 background "
                "details (demographic, academic, social, behavioural) improved the cross-validated "
                "accuracy once G1 was known, so the form does not ask for them. The Experiments "
                "page shows what adding them back does.")
        return student
    for group, tab in zip(settings.GROUPS, st.tabs(settings.GROUPS)):
        with tab:
            in_group = [field for field in form if field["group"] == group and field["column"] != "G1"]
            if not in_group:
                st.caption(f"Feature selection found no {group.lower()} detail that improves "
                           f"the prediction, so none is asked.")
            for field in in_group:
                student[field["column"]] = ask(field)
    return student


def verdict(probabilities: dict[str, float]) -> str:
    """The sentence under the status, as sure as the probability allows."""
    ranked = sorted(probabilities, key=probabilities.get, reverse=True)
    top, second = ranked[0], ranked[1]
    index = settings.CATEGORIES.index(top)
    if probabilities[top] >= SURE:
        sentence = f"Likely {top.lower()}: final grade {grade_range(index)}."
    elif probabilities[top] >= PROBABLE:
        sentence = f"Probably {top.lower()} (final grade {grade_range(index)}), but not certain."
    else:
        sentence = (f"Unclear: between {top.lower()} ({probabilities[top]:.0%}) and "
                    f"{second.lower()} ({probabilities[second]:.0%}).")
    return f"{sentence} {ADVICE[index]}".strip()


def status_block(probabilities: dict[str, float], predicted: str) -> None:
    """The predicted group in its lamp colour, with the sentence that goes with it."""
    index = settings.CATEGORIES.index(predicted)
    st.html(f"""
        <div class="status" style="--status-colour:{GROUP_COLOURS[index]}">
          <div class="label">Predicted final group</div>
          <div class="group">{predicted}</div>
          <div class="verdict">{verdict(probabilities)}</div>
        </div>""")


def chance_meters(probabilities: dict[str, float]) -> None:
    """The headline risk meter (green -> amber -> red), then one meter per group."""
    support = probabilities[settings.CATEGORIES[SUPPORT]]
    rows = [f"""<div class="meter-row headline">
                  <div class="name">Chance of needing support<br>
                    <span class="range">final grade below 10</span></div>
                  {meter(support, risk_colour)}
                  <div class="value">{support:.0%}</div></div>"""]
    for index, (group, chance) in enumerate(probabilities.items()):
        rows.append(f"""<div class="meter-row">
                          <div class="name">{group_lamp(index)}{group}
                            <span class="range">{grade_range(index)}</span></div>
                          {meter(chance, lambda position, colour=GROUP_COLOURS[index]: colour)}
                          <div class="value">{chance:.0%}</div></div>""")
    st.html("".join(rows))
    st.caption("Each meter has 20 cells of 5 %. The chances are calibrated on validation "
               "students, so 70 % means right about 7 times in 10.")


def describe(field: dict, value) -> str:
    """An answer in words: 'Three or more' instead of 3."""
    return str(dict((key, label) for key, label in field["choices"])[value]) if "choices" in field else str(value)


def show_reasons(predictor, student: dict, predicted: str) -> None:
    effects = predictor.explain(student)
    st.subheader("What pushed this prediction", anchor=False)
    if not effects:
        st.write("Every answer is the most common one, so nothing stands out.")
        return
    fields = {field["column"]: field for field in predictor.form}
    lines = []
    for effect in effects[:5]:
        field = fields[effect["column"]]
        direction = "towards" if effect["effect"] > 0 else "away from"
        lines.append(f"- **{field['question']}: {describe(field, effect['answer'])}** "
                     f"(typical: {describe(field, effect['typical'])}) moves it "
                     f"{abs(effect['effect']):.0%} {direction} {predicted}")
    st.markdown("\n".join(lines))
    st.caption("Each answer is set back to the typical one, one at a time, and the change in "
               "probability is shown. Answers can also work together, which this does not show.")


start_page("Check a student")
predictor = load_predictor()
st.write("For use after the first term. Start with the student's first-term grade; the form "
         "begins on the most common answer for everything else.")

inputs_column, result_column = st.columns([2, 3], gap="large")
with inputs_column:
    st.subheader("Student", anchor=False)
    student = student_inputs(predictor.form)
with result_column:
    probabilities = predictor.predict(student)
    predicted = max(probabilities, key=probabilities.get)
    status_block(probabilities, predicted)
    chance_meters(probabilities)
    show_reasons(predictor, student, predicted)
