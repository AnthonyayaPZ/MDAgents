import os
import json
import base64
import random
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from openai import OpenAI
from pptree import *


# ---------------------------------------------------------------------------
# Structured JSON schema for the AD treatment plan output
# ---------------------------------------------------------------------------
AD_TREATMENT_PLAN_SCHEMA = {
    "patient_summary": {
        "age": "",
        "sex": "",
        "chief_complaint": "",
        "relevant_history": [],
        "comorbidities": []
    },
    "diagnosis": {
        "primary_diagnosis": "",
        "confidence_level": "",
        "differential_diagnoses": [],
        "supporting_evidence": {
            "clinical_findings": [],
            "neuroimaging_findings": [],
            "cognitive_assessment": []
        }
    },
    "disease_staging": {
        "stage": "",
        "staging_criteria": "",
        "functional_status": "",
        "cognitive_domain_impairments": []
    },
    "pharmacological_interventions": {
        "cognitive_enhancers": [],
        "behavioral_symptom_management": [],
        "comorbidity_management": [],
        "medications_to_avoid": []
    },
    "non_pharmacological_interventions": {
        "cognitive_rehabilitation": [],
        "physical_activity_plan": [],
        "occupational_therapy": [],
        "speech_language_therapy": [],
        "nutritional_recommendations": [],
        "sleep_hygiene": []
    },
    "caregiver_support": {
        "education_resources": [],
        "respite_care_recommendations": [],
        "support_groups": [],
        "safety_modifications": []
    },
    "monitoring_plan": {
        "cognitive_reassessment_schedule": "",
        "neuroimaging_follow_up": "",
        "laboratory_monitoring": [],
        "medication_review_schedule": ""
    },
    "follow_up_plan": {
        "next_visit": "",
        "referrals": [],
        "goals_of_care": [],
        "advance_care_planning": ""
    }
}


# ---------------------------------------------------------------------------
# Helper: encode a local image file to base64 for the OpenAI vision API
# ---------------------------------------------------------------------------
def encode_image_to_base64(image_path: str) -> str:
    """Read a local image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Agent – mirrors the original but adds multi-modal (image) support
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, instruction, role, examplers=None,
                 model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro':
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = OpenAI(api_key=os.environ['openai_api_key'])
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append(
                        {"role": "user", "content": exampler['question']})
                    self.messages.append(
                        {"role": "assistant",
                         "content": exampler['answer'] + "\n\n" + exampler['reason']})

    # ---- core chat (supports optional image) ---------------------------------
    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    return responses
                except Exception:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            # Build content (text-only or multimodal)
            if img_path and self.model_info in ['gpt-4o', 'gpt-4o-mini']:
                b64 = encode_image_to_base64(img_path)
                content = [
                    {"type": "text", "text": message},
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/png;base64,{b64}"
                     }}
                ]
            else:
                content = message

            self.messages.append({"role": "user", "content": content})

            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "gpt-4o-mini"

            response = self.client.chat.completions.create(
                model=model_name,
                messages=self.messages
            )
            assistant_msg = response.choices[0].message.content
            self.messages.append(
                {"role": "assistant", "content": assistant_msg})
            return assistant_msg

    # ---- temperature sweep for final decision --------------------------------
    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            if img_path and self.model_info in ['gpt-4o', 'gpt-4o-mini']:
                b64 = encode_image_to_base64(img_path)
                content = [
                    {"type": "text", "text": message},
                    {"type": "image_url",
                     "image_url": {
                         "url": f"data:image/png;base64,{b64}"
                     }}
                ]
            else:
                content = message

            self.messages.append({"role": "user", "content": content})

            temperatures = [0.0]
            responses = {}
            model_name = ('gpt-3.5-turbo'
                          if self.model_info == 'gpt-3.5' else 'gpt-4o-mini')
            for temperature in temperatures:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    temperature=temperature,
                )
                responses[temperature] = response.choices[0].message.content
            return responses

        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            return responses


# ---------------------------------------------------------------------------
# Group – AD-specific multi-agent group with internal collaboration
# ---------------------------------------------------------------------------
class Group:
    def __init__(self, goal, members, clinical_input, img_path=None):
        self.goal = goal
        self.members = []
        self.img_path = img_path
        for member_info in members:
            _agent = Agent(
                'You are a {} who {}. You are part of a multidisciplinary '
                'team evaluating a patient with suspected or confirmed '
                'Alzheimer\'s disease.'.format(
                    member_info['role'],
                    member_info['expertise_description'].lower()),
                role=member_info['role'],
                model_info='gpt-4o-mini')
            _agent.chat(
                'You are a {} who {}. You are part of a multidisciplinary '
                'team evaluating a patient with suspected or confirmed '
                'Alzheimer\'s disease.'.format(
                    member_info['role'],
                    member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.clinical_input = clinical_input

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                if 'lead' in member.role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]

            delivery_prompt = (
                f"You are the lead of a clinical team whose goal is to "
                f"{self.goal} for a patient with suspected Alzheimer's "
                f"disease. Your assistant clinicians are:")
            for a_mem in assist_members:
                delivery_prompt += f"\n- {a_mem.role}"

            delivery_prompt += (
                "\n\nGiven the following clinical information, specify what "
                "investigations or analyses each assistant should perform.\n\n"
                f"Clinical Input:\n{self.clinical_input}")

            img = img_path or self.img_path
            try:
                delivery = lead_member.chat(delivery_prompt, img_path=img)
            except Exception:
                delivery = assist_members[0].chat(
                    delivery_prompt, img_path=img)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat(
                    "Your team goal is: {}. The team lead requests the "
                    "following investigations:\n{}\n\nPlease provide your "
                    "investigation summary focused on Alzheimer's disease "
                    "assessment and treatment planning.".format(
                        self.goal, delivery))
                investigations.append([a_mem.role, investigation])

            gathered = ""
            for inv in investigations:
                gathered += f"[{inv[0]}]\n{inv[1]}\n"

            investigation_prompt = (
                f"The gathered investigations from your team:\n{gathered}\n\n"
                f"Clinical Input:\n{self.clinical_input}\n\n"
                "Now synthesize the team's findings and provide your "
                "assessment relevant to the Alzheimer's disease treatment "
                "plan.")

            response = lead_member.chat(investigation_prompt)
            return response

        elif comm_type == 'external':
            return


# ---------------------------------------------------------------------------
# Hierarchy / group info parsing (unchanged from original)
# ---------------------------------------------------------------------------
def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except Exception:
            expert = expert.split('-')[0].strip()
        if hierarchy is None:
            hierarchy = 'Independent'
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()
            for agent in agents:
                if (agent.name.split("(")[0].strip().lower()
                        == parent.strip().lower()):
                    child_agent = Node(
                        "{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)
        else:
            agent = Node(
                "{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)
        count += 1
    return agents


def parse_group_info(group_info):
    lines = group_info.split('\n')
    parsed_info = {
        'group_goal': '',
        'members': []
    }
    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            member_role = member_role_description[0].strip()
            member_expertise = (member_role_description[1].strip()
                                if len(member_role_description) > 1 else '')
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    return parsed_info


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        client = OpenAI(api_key=os.environ['openai_api_key'])
        return None, client
    else:
        raise ValueError(f"Unsupported model: {model_name}")


# ---------------------------------------------------------------------------
# Build the clinical input string from a text report + optional MRI path
# ---------------------------------------------------------------------------
def build_clinical_input(text_report, img_path=None):
    """Combine the text report and image path into a clinical context string."""
    clinical_input = f"=== Clinical Text Report ===\n{text_report}\n"
    if img_path:
        clinical_input += (
            f"\n=== Brain MRI Image ===\n"
            f"[An MRI image has been provided at: {img_path}. "
            f"It will be passed to vision-capable models for analysis.]\n")
    return clinical_input


# ---------------------------------------------------------------------------
# Determine complexity / difficulty of the AD case
# ---------------------------------------------------------------------------
def determine_difficulty(clinical_input, difficulty, img_path=None):
    if difficulty != 'adaptive':
        return difficulty

    difficulty_prompt = (
        "You are a senior neurologist performing triage on an Alzheimer's "
        "disease case. Based on the clinical information below, classify the "
        "case complexity.\n\n"
        f"{clinical_input}\n\n"
        "Classify the case into ONE of the following levels:\n"
        "1) basic: The case is straightforward — typical presentation, clear "
        "diagnosis, standard treatment. A single specialist can create the "
        "treatment plan.\n"
        "2) intermediate: The case has moderate complexity — atypical features, "
        "multiple comorbidities, or medication interactions requiring input "
        "from several specialists.\n"
        "3) advanced: The case is highly complex — young-onset AD, mixed "
        "dementia, severe behavioral symptoms, complex ethical considerations, "
        "or multi-organ involvement requiring full multidisciplinary teams.\n\n"
        "Return ONLY one word: basic, intermediate, or advanced."
    )

    triage_agent = Agent(
        instruction=(
            'You are a senior neurologist who triages neurodegenerative '
            'disease cases and determines their complexity level.'),
        role='triage neurologist',
        model_info='gpt-3.5')
    triage_agent.chat(
        'You are a senior neurologist who triages neurodegenerative '
        'disease cases and determines their complexity level.')
    response = triage_agent.chat(difficulty_prompt, img_path=img_path)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'
    return 'intermediate'  # safe default


# ---------------------------------------------------------------------------
# JSON output formatting prompt (shared across all difficulty paths)
# ---------------------------------------------------------------------------
TREATMENT_PLAN_JSON_INSTRUCTION = """
Return the treatment plan as a valid JSON object using EXACTLY the following
structure.  Fill every field with clinically appropriate content based on
the patient's information.  If a field is not applicable write "N/A".

```json
{schema}
```

IMPORTANT: Return ONLY the JSON object, no extra commentary.
""".format(schema=json.dumps(AD_TREATMENT_PLAN_SCHEMA, indent=2))


# ---------------------------------------------------------------------------
# BASIC: single-specialist pathway
# ---------------------------------------------------------------------------
def process_basic_query(clinical_input, model, img_path=None):
    cprint("[INFO] Basic pathway — Single Specialist Assessment",
           'yellow', attrs=['blink'])

    specialist = Agent(
        instruction=(
            "You are a board-certified neurologist specializing in "
            "Alzheimer's disease and related dementias. You have extensive "
            "experience creating individualized treatment plans for AD "
            "patients. When provided with a brain MRI image, analyze it for "
            "patterns of cortical atrophy, hippocampal volume loss, "
            "ventricular enlargement, and white matter changes consistent "
            "with neurodegenerative disease."),
        role='AD specialist neurologist',
        model_info=model)
    specialist.chat(
        "You are a board-certified neurologist specializing in Alzheimer's "
        "disease and related dementias.")

    assessment = specialist.chat(
        f"Please review the following clinical information for a patient "
        f"with suspected or confirmed Alzheimer's disease and perform a "
        f"comprehensive assessment.\n\n{clinical_input}\n\n"
        f"Provide:\n"
        f"1. Your diagnostic impression and confidence level\n"
        f"2. Disease staging\n"
        f"3. Recommended pharmacological and non-pharmacological interventions\n"
        f"4. Monitoring and follow-up plan",
        img_path=img_path)

    cprint("[INFO] Generating structured treatment plan …",
           'yellow', attrs=['blink'])

    plan = specialist.temp_responses(
        f"Based on your assessment:\n{assessment}\n\n"
        f"{TREATMENT_PLAN_JSON_INSTRUCTION}",
        img_path=img_path)

    return plan


# ---------------------------------------------------------------------------
# INTERMEDIATE: multi-expert discussion pathway
# ---------------------------------------------------------------------------
def process_intermediate_query(clinical_input, model, img_path=None):
    cprint("[INFO] Intermediate pathway — Multi-Specialist Discussion",
           'yellow', attrs=['blink'])

    # Step 1: Recruit specialists
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = (
        "You are a senior neurologist assembling a specialist panel to "
        "develop a treatment plan for a patient with Alzheimer's disease. "
        "You must recruit experts whose combined expertise covers diagnosis "
        "confirmation, neuroimaging interpretation, pharmacotherapy, "
        "non-pharmacological interventions, and caregiver support.")

    recruiter = Agent(
        instruction=recruit_prompt, role='recruiter', model_info='gpt-3.5')
    recruiter.chat(recruit_prompt)

    num_agents = 5
    recruited = recruiter.chat(
        f"Clinical Information:\n{clinical_input}\n\n"
        f"Recruit {num_agents} experts to collaboratively create an AD "
        f"treatment plan. For each expert provide their specialty, "
        f"a brief description, and their communication hierarchy.\n\n"
        f"Format each expert as:\n"
        f"1. Specialty - Description of expertise. - Hierarchy: Independent\n"
        f"2. Specialty - Description. - Hierarchy: Specialty1 > Specialty2\n\n"
        f"Example:\n"
        f"1. Neurologist - Leads the diagnostic workup and oversees the "
        f"overall treatment plan for neurodegenerative diseases. "
        f"- Hierarchy: Independent\n"
        f"2. Neuroradiologist - Interprets brain MRI scans for atrophy "
        f"patterns and vascular changes. - Hierarchy: Neurologist > "
        f"Neuroradiologist\n"
        f"3. Geriatric Psychiatrist - Manages behavioral and psychological "
        f"symptoms of dementia. - Hierarchy: Independent\n"
        f"4. Geriatrician - Addresses comorbidities, polypharmacy, and "
        f"overall geriatric care. - Hierarchy: Independent\n"
        f"5. Neuropsychologist - Administers and interprets cognitive "
        f"assessments. - Hierarchy: Independent\n\n"
        f"Return ONLY the numbered list, no extra text.")

    agents_info = [
        agent_info.split(" - Hierarchy: ")
        for agent_info in recruited.split('\n') if agent_info.strip()
    ]
    agents_data = [
        (info[0], info[1]) if len(info) > 1 else (info[0], None)
        for info in agents_info
    ]

    agent_emoji = [
        '\U0001F468\u200D\u2695\uFE0F',
        '\U0001F468\U0001F3FB\u200D\u2695\uFE0F',
        '\U0001F469\U0001F3FC\u200D\u2695\uFE0F',
        '\U0001F469\U0001F3FB\u200D\u2695\uFE0F',
        '\U0001f9d1\u200D\u2695\uFE0F',
        '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F',
        '\U0001f468\U0001f3ff\u200D\u2695\uFE0F',
        '\U0001f468\U0001f3fd\u200D\u2695\uFE0F',
        '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F',
        '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    agent_dict = {}
    medical_agents = []
    for i, agent_data in enumerate(agents_data):
        try:
            agent_role = agent_data[0].split('-')[0].split('.')[1].strip().lower()
            description = agent_data[0].split('-')[1].strip().lower()
        except Exception:
            continue

        inst = (
            f"You are a {agent_role} who {description}. "
            f"You are collaborating with other specialists to create a "
            f"comprehensive treatment plan for a patient with Alzheimer's "
            f"disease. When reviewing neuroimaging, look for hallmarks of AD "
            f"including medial temporal lobe atrophy, posterior cortical "
            f"atrophy, and patterns of neurodegeneration.")
        _agent = Agent(instruction=inst, role=agent_role, model_info=model)
        _agent.chat(inst)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    for idx, agent_data in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} "
                  f"{agent_data[0].split('-')[0].strip()}): "
                  f"{agent_data[0].split('-')[1].strip()}")
        except Exception:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent_data[0]}")

    # Step 2: Collaborative deliberation
    print()
    cprint("[INFO] Step 2. Collaborative Deliberation",
           'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection",
           'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 3
    num_turns = 3
    num_agents_count = len(medical_agents)

    interaction_log = {
        f'Round {r}': {
            f'Turn {t}': {
                f'Agent {s}': {
                    f'Agent {tgt}': None
                    for tgt in range(1, num_agents_count + 1)
                } for s in range(1, num_agents_count + 1)
            } for t in range(1, num_turns + 1)
        } for r in range(1, num_rounds + 1)
    }

    cprint("[INFO] Step 2.2. Specialist Consultation",
           'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds + 1)}
    for k, v in agent_dict.items():
        opinion = v.chat(
            f"Review the following clinical information for a patient with "
            f"suspected Alzheimer's disease. Provide your specialist opinion "
            f"on diagnosis, severity, and treatment recommendations within "
            f"your area of expertise.\n\n{clinical_input}\n\n"
            f"Focus your response on your area of specialty.",
            img_path=img_path)
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds + 1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        synthesizer = Agent(
            instruction=(
                "You are a clinical coordinator who synthesizes opinions "
                "from multiple AD specialists into a coherent treatment "
                "plan."),
            role="clinical coordinator",
            model_info=model)
        synthesizer.chat(
            "You are a clinical coordinator who synthesizes opinions "
            "from multiple AD specialists.")

        assessment = "".join(
            f"({k}): {v}\n" for k, v in round_opinions[n].items())

        synthesizer.chat(
            f"Here are specialist opinions for an AD patient:\n\n"
            f"{assessment}\n\n"
            f"Synthesize the key findings and identify areas of agreement "
            f"and disagreement relevant to treatment planning.")

        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(
                    f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n"
                    for _k, _v in interaction_log[round_name][turn_name].items())

                participate = v.chat(
                    "Given the specialist opinions, do you want to discuss "
                    "with any other expert to refine the AD treatment plan? "
                    "(yes/no)\n\nOpinions:\n{}".format(
                        assessment if n == 1 else all_comments))

                if 'yes' in participate.lower().strip():
                    chosen_expert = v.chat(
                        f"Which expert do you want to consult?\n{agent_list}\n"
                        f"Return just the number(s), e.g. 1 or 1,2")

                    chosen_experts = [
                        int(ce) for ce in
                        chosen_expert.replace('.', ',').split(',')
                        if ce.strip().isdigit()
                    ]

                    for ce in chosen_experts:
                        if ce < 1 or ce > len(medical_agents):
                            continue
                        specific_question = v.chat(
                            f"Share your specialist opinion with Agent {ce} "
                            f"({medical_agents[ce-1].role}) regarding the AD "
                            f"treatment plan. Be specific and concise.")

                        print(f" Agent {idx+1} ({agent_emoji[idx]} "
                              f"{medical_agents[idx].role}) -> Agent {ce} "
                              f"({agent_emoji[ce-1]} "
                              f"{medical_agents[ce-1].role})")
                        interaction_log[round_name][turn_name][
                            f'Agent {idx+1}'][f'Agent {ce}'] = specific_question

                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} "
                          f"{v.role}): \U0001f910")

            if num_yes == 0:
                break

        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(
                "After discussion with other specialists, provide your final "
                "recommendations for this AD patient's treatment plan, "
                "focusing on your specialty area.\n\n"
                f"Clinical Information:\n{clinical_input}")
            tmp_final_answer[agent.role] = response

        final_answer = tmp_final_answer

    # Interaction log table
    print('\nInteraction Log')
    myTable = PrettyTable(
        [''] + [f"Agent {i+1} ({agent_emoji[i]})"
                for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents) + 1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents) + 1):
            if i == j:
                row.append('')
            else:
                i2j = any(
                    interaction_log[f'Round {k}'][f'Turn {l}'][
                        f'Agent {i}'][f'Agent {j}'] is not None
                    for k in range(1, len(interaction_log) + 1)
                    for l in range(1, len(interaction_log['Round 1']) + 1))
                j2i = any(
                    interaction_log[f'Round {k}'][f'Turn {l}'][
                        f'Agent {j}'][f'Agent {i}'] is not None
                    for k in range(1, len(interaction_log) + 1)
                    for l in range(1, len(interaction_log['Round 1']) + 1))

                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents) + 1)])

    print(myTable)

    # Step 3: Final treatment plan generation
    cprint("\n[INFO] Step 3. Final Treatment Plan Generation",
           'yellow', attrs=['blink'])

    moderator = Agent(
        "You are the lead neurologist who synthesizes all specialist "
        "opinions into a final, structured AD treatment plan. You have "
        "expertise in evidence-based management of Alzheimer's disease.",
        "Treatment Plan Moderator",
        model_info=model)
    moderator.chat(
        "You are the lead neurologist finalizing the AD treatment plan.")

    _decision = moderator.temp_responses(
        f"Below are the final recommendations from each specialist:\n\n"
        f"{json.dumps(final_answer, indent=2) if final_answer else 'No specialist recommendations available.'}\n\n"
        f"Original clinical information:\n{clinical_input}\n\n"
        f"Synthesize all recommendations into a comprehensive treatment plan."
        f"\n\n{TREATMENT_PLAN_JSON_INSTRUCTION}",
        img_path=img_path)

    return _decision


# ---------------------------------------------------------------------------
# ADVANCED: full MDT (Multidisciplinary Team) pathway
# ---------------------------------------------------------------------------
def process_advanced_query(clinical_input, model, img_path=None):
    cprint("[INFO] Advanced pathway — Full MDT Collaboration",
           'yellow', attrs=['blink'])

    # Step 1: Recruit MDTs
    print("[STEP 1] MDT Recruitment")
    group_instances = []

    recruit_prompt = (
        "You are a senior neurologist organizing Multidisciplinary Teams "
        "(MDTs) to develop a comprehensive treatment plan for a complex "
        "Alzheimer's disease case. Each team should have a specific focus "
        "area relevant to AD management.")

    recruiter = Agent(
        instruction=recruit_prompt, role='recruiter', model_info='gpt-4o-mini')
    recruiter.chat(recruit_prompt)

    num_teams = 4
    num_members = 3

    recruited = recruiter.chat(
        f"Clinical Information:\n{clinical_input}\n\n"
        f"Organize {num_teams} MDTs with {num_members} clinicians each to "
        f"develop a treatment plan for this AD patient.\n\n"
        f"You MUST include:\n"
        f"- Initial Assessment Team (IAT): Confirms diagnosis and stages "
        f"the disease\n"
        f"- Final Review and Decision Team (FRDT): Reviews all team outputs "
        f"and produces the final plan\n\n"
        f"Example format:\n"
        f"Group 1 - Initial Assessment Team (IAT)\n"
        f"Member 1: Neurologist (Lead) - Leads the diagnostic workup "
        f"including clinical history, neurological exam, and cognitive "
        f"testing. Determines disease stage using established criteria.\n"
        f"Member 2: Neuroradiologist - Interprets brain MRI for hippocampal "
        f"atrophy, cortical thinning, white matter hyperintensities, and "
        f"other neurodegenerative markers.\n"
        f"Member 3: Neuropsychologist - Administers and interprets "
        f"standardized cognitive assessments (MMSE, MoCA, neuropsychological "
        f"battery) to characterize the cognitive profile.\n\n"
        f"Group 2 - Pharmacotherapy Team (PT)\n"
        f"Member 1: Neurologist (Lead) - Prescribes and monitors "
        f"cholinesterase inhibitors, memantine, and emerging AD therapies.\n"
        f"Member 2: Geriatric Psychiatrist - Manages behavioral and "
        f"psychological symptoms of dementia (BPSD) with appropriate "
        f"psychotropic medications.\n"
        f"Member 3: Clinical Pharmacist - Reviews drug interactions, "
        f"optimizes medication regimen, and monitors for adverse effects.\n\n"
        f"Group 3 - Non-Pharmacological and Support Team (NPST)\n"
        f"Member 1: Geriatrician (Lead) - Coordinates non-pharmacological "
        f"interventions and addresses comorbid conditions.\n"
        f"Member 2: Occupational Therapist - Develops strategies for "
        f"maintaining activities of daily living and home safety.\n"
        f"Member 3: Social Worker - Coordinates caregiver support, community "
        f"resources, and advance care planning.\n\n"
        f"Group 4 - Final Review and Decision Team (FRDT)\n"
        f"Member 1: Senior Neurologist (Lead) - Provides overarching "
        f"clinical expertise and ensures evidence-based recommendations.\n"
        f"Member 2: Clinical Decision Specialist - Integrates all team "
        f"recommendations into a unified treatment plan.\n"
        f"Member 3: Palliative Care Specialist - Ensures goals-of-care "
        f"discussions and advance care planning are addressed.\n\n"
        f"Please strictly follow the above format.")

    groups = [g.strip() for g in recruited.split("Group") if g.strip()]
    group_strings = ["Group " + g for g in groups]

    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): "
                  f"{member['expertise_description']}")
        print()

        group_instance = Group(
            res_gs['group_goal'], res_gs['members'],
            clinical_input, img_path=img_path)
        group_instances.append(group_instance)

    # Step 2: Team assessments
    # 2.1 Initial Assessment Team
    print("[STEP 2] Team Assessments")
    initial_assessments = []
    for gi in group_instances:
        if ('initial' in gi.goal.lower() or 'iat' in gi.goal.lower()):
            init_assessment = gi.interact(
                comm_type='internal', img_path=img_path)
            initial_assessments.append([gi.goal, init_assessment])

    initial_report = ""
    for idx, ia in enumerate(initial_assessments):
        initial_report += f"Group {idx+1} - {ia[0]}\n{ia[1]}\n\n"

    # 2.2 Other MDTs
    assessments = []
    for gi in group_instances:
        if ('initial' not in gi.goal.lower()
                and 'iat' not in gi.goal.lower()
                and 'review' not in gi.goal.lower()
                and 'decision' not in gi.goal.lower()
                and 'frdt' not in gi.goal.lower()):
            assessment = gi.interact(comm_type='internal', img_path=img_path)
            assessments.append([gi.goal, assessment])

    assessment_report = ""
    for idx, a in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {a[0]}\n{a[1]}\n\n"

    # 2.3 Final Review and Decision Team
    final_decisions = []
    for gi in group_instances:
        if ('review' in gi.goal.lower()
                or 'decision' in gi.goal.lower()
                or 'frdt' in gi.goal.lower()):
            decision = gi.interact(comm_type='internal', img_path=img_path)
            final_decisions.append([gi.goal, decision])

    compiled_report = ""
    for idx, d in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {d[0]}\n{d[1]}\n\n"

    # Step 3: Final structured treatment plan
    print("[STEP 3] Final Treatment Plan Generation")
    decision_prompt = (
        "You are a senior neurologist and AD specialist. You have received "
        "assessments from multiple multidisciplinary teams. Your task is to "
        "produce a final, comprehensive, structured treatment plan for the "
        "patient's Alzheimer's disease.")

    final_agent = Agent(
        instruction=decision_prompt, role='decision maker', model_info=model)
    final_agent.chat(decision_prompt)

    final_plan = final_agent.temp_responses(
        f"=== Initial Assessment ===\n{initial_report}\n\n"
        f"=== Specialist Team Reports ===\n{assessment_report}\n\n"
        f"=== Final Review Team Report ===\n{compiled_report}\n\n"
        f"=== Original Clinical Information ===\n{clinical_input}\n\n"
        f"Now produce the final treatment plan.\n\n"
        f"{TREATMENT_PLAN_JSON_INSTRUCTION}",
        img_path=img_path)

    return final_plan
