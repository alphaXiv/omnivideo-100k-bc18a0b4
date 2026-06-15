import os
import re
import time
import json
import random
import asyncio
import argparse
import aiofiles
from tqdm import tqdm
from google import genai


fine_grained_perception_prompt = """
# Role
You are an expert in fine-grained multi-modal perception, specializing in dissecting and describing the intricate interplay between subtle auditory and visual details in complex video scenes.

# Task
Analyze the provided textual description of a video (including a summary, main entities, and detailed script) to generate two fine-grained perception Q&A pairs based on subtle, non-salient events.
You will generate two Q&A pairs:
- Audio-Guided Visual Description: One question that uses a subtle sound as a cue and asks for a detailed description of a concurrent visual event.
- Vision-Guided Audio Description: One question that uses a subtle visual event as a cue and asks for a detailed description of the concurrent sounds.
Each question must hinge on a detail that is easily missed and requires the explicit correlation of audio and visual information to answer correctly.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}

# Instructions
1. Identify Unique, Subtle Inter-Modal Cues:
Meticulously scan the entire video description to find moments where a subtle, non-salient event in one modality (the cue) occurs simultaneously with a specific, clearly defined event in the other modality. 
2. Verify Uniqueness of the Cue:
For any candidate cue (the subtle audio or visual event), you must confirm that it is unique within the video to ensure the question is unambiguous. If the cue is not unique, you must find a different one.
3. Handle Missing Events:
If, after a thorough search, a suitable unique and subtle cue cannot be found for either the "Audio-Guided" or "Vision-Guided" category, you must write NONE under that category's heading and provide no Q, A, or Analysis for it.
4. Formulate Perception Questions:
Based on the unique and subtle cues you've identified, craft:
- An audio-guided question that describes a specific, subtle sound and asks what is visually happening at that exact moment.
- A vision-guided question that describes a specific, subtle visual action and asks for a detailed description of the sounds occurring simultaneously.
5. Provide Descriptive Answers:
For each question, provide a concise but detailed sentence that accurately describes the event in the target modality. The answer should focus on the specific details that co-occur with the cue provided in the question.

# Constraints:
1. Question must provide a subtle cue from one modality (audio or visual) and the answer must describe a simultaneous event from the other modality. The connection between the cue and the described event must require careful temporal correlation.
2. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
3. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Audio-Guided
Q: [Your question describing a subtle sound and asking for a visual description]
A: [A detailed sentence describing the concurrent visual event]
Analysis: [Explain why this question requires correlating the specific audio cue with the visual scene and what subtle detail it is based on]
Vision-Guided
Q: [Your question describing a subtle visual event and asking for an audio description]
A: [A detailed sentence describing the concurrent soundscape/specific sounds]
Analysis: [Explain why this question requires correlating the specific visual cue with the audio track and what subtle detail it is based on]
""".strip()


scene_transformation_detection_prompt = """
# Role
You are an expert in multi-modal scene analysis, specializing in using audio events as temporal markers to identify and describe concurrent visual changes in a video.

# Task
Analyze the provided textual description of a video (including a summary, main entities, and detailed script) to generate up to two precise Q&A pairs. The task is to identify a significant visual scene change that occurs during a specific, identifiable audio event which serves as a temporal clue.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}

# Instructions
1. Identify Defining Audio Events:
Scan the script to find audio events that have a clear start and end. These events will serve as temporal markers or "audio contexts."
2. Find a Visual Change within the Audio Context:
Look for a significant visual change that occurs while one of the identified audio events is in progress. A visual change can be either a transformation in location/setting or a significant shift in camera perspective.
3. Verify Uniqueness:
Ensure that the audio event described in the question is specific enough to refer to a single, unambiguous period in the video.
4. Handle Missing Events:
If, after a thorough search, no suitable visual scene transformations occurring within a definable audio context can be found, you must return the single word NONE and nothing else.
5. Formulate Questions:
Craft a natural language question that uses the audio event as a temporal clue to ask about the concurrent change in the visual scene.
6. Provide Precise Answers:
Provide a clear, descriptive sentence that explains the visual change, whether it's a change in location or perspective.

# Constraints:
1. Question must be framed around an audio event that provides the temporal context for the visual change.
2. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
3. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q1: [Your natural language question using an audio event as a temporal context]
A1: [A descriptive sentence explaining the visual change.]
Analysis: [Explain what the defining audio event is and how it serves as a temporal marker for a separate visual change, requiring the user to process both modalities to answer correctly.]
Q2: [Your natural language question using an audio event as a temporal context]
A2: [A descriptive sentence explaining the visual change.]
Analysis: [Explain what the defining audio event is and how it serves as a temporal marker for a separate visual change, requiring the user to process both modalities to answer correctly.]
""".strip()


context_understanding_prompt = """
# Role
You are an expert in Multimodal Context Understanding. Your expertise lies in identifying Strong Multimodal Correlations where the Audio and Visual streams complement each other to tell a fuller story.

# Task
Analyze the provided textual description of a video (including a summary, main entities, and detailed script) to generate two high-quality "Context Understanding" Q&A pairs.
You will generate two Q&A pairs:
- Visual Context: One question that cites a specific verbal mention (quote, plan, or concept) or sound and asks for the related visual context.
- Audio Context: One question that cites a specific visual action or scene and asks for the explanatory context found in the dialogue or audio.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}

# Instructions
1. Pinpoint Contextual Anchor:
Meticulously scan the entire video description to identify specific triggers that act as a focal point for inquiry.
- Look for Audio Anchors: Specific mentions of people, abstract concepts, plans, or emotions.
- Look for Visual Anchors: Distinct actions, complex scenarios, or specific interactions that might lack definition without audio context.
2. Retrieve Complementary Evidence:
For each identified Anchor, search the opposing modality (Visuals for Audio anchors; Audio for Visual anchors) to find the specific details that flesh out the story.
3. Verify Information Gain:
Strictly filter the pairs. You must ensure the connection is Non-Trivial.
- Reject: pairs where the visual is a mere repetition of the text (e.g., saying "apple" while holding an apple).
- Keep: pairs only where the Context provides new information
4. Handle Missing Pairs:
If, after a thorough search, a suitable unique and subtle cue cannot be found for either the "Visual Context" or "Audio Context" category, you must write NONE under that category's heading and provide no Q, A, or Analysis for it.
5. Formulate Context-Seeking Question:
Based on the Contextual Anchor and Complementary Evidence pairs you've identified, craft:
- A Visual Context question that clearly cite the spoken phrase, name, or sound and ask for the related visual manifestation.
- A Audio Context question that clearly describe the visual action or scene and ask for the informational context provided by the dialogue/audio.
6. Provide Descriptive Answers:
For each question, provide a concise but detailed sentence that accurately describes the information in the target modality.

# Constraints
1. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
2. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Visual Context
Q: [Your question citing a audio information and asking for visual context]
A: [A detailed sentence describing the visual evidence/setting]
Analysis: [Explain the "Information Gain": specifically what visual details ground the verbal reference]
Audio Context
Q: [Your question citing a visual information and asking for audio context]
A: [A detailed sentence describing the situation/reason found in the audio]
Analysis: [Explain the "Information Gain": specifically how the audio explains or defines the visual action]
""".strip()


comparison_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Differentiation and Nuanced Analysis. Your expertise lies in distinguishing between distinct but related concepts, actions, states or events by synthesizing audio and visual cues.

# Task
Analyze the provided detailed video description to exhaustively uncover nuanced distinctions. You will do this by identifying groups of video segments where two entities or events are conceptually related but differ in specific aspect. You must demonstrate exactly what those differences are by synthesizing audio and visual cues.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Related but Distinct Subjects:
Meticulously scan the entire video description for pairs of events or entities that share a common theme but represent different variations. Avoid obvious, unrelated opposites (like "Cat vs. Car").
2. Pinpoint the Nuance:
Determine exactly what the subtle differentiator is.
3. Group and Link:
List the relevant segments. You must ensure that the comparison relies on combining audio descriptions from one part with visual evidence from another.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct, well-supported comparative distinctions capable of being inferred from the video..
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues from one or more segments and visual evidence from another one or more segments interact to help understanding the nuance.]
""".strip()


comparison_prompt_2 = """
# Role
You are an expert analyst specializing in distinguishing fine details in multimodal media. Your expertise lies in probing the subtle differences or specific nuances between two related elements in a video.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a single, insightful Q&A pair focused on comparison.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Understand the Nuance:
Examine the Designated Segments, referring to the provided connections, identify the common thread connecting the two subjects and the specific aspect that differentiates them.
2. Formulate a Precise Comparative Question:
Craft a "How does differ from" question that acknowledges the relation but asks for the distinction.
3. Provide Integrated Answer:
In the answer, explicitly describe the nuance. You must cite evidence from the different designated segments-both visual and auditory- to explain how they differ in that specific aspect.

# Constraints:
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question]
A: [Your Answer]
""".strip()


sentiment_analysis_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Sentiment Analysis. Your expertise lies in decoding complex human internal states by synthesizing disparate elements-dialogue, sounds, actions, and visual cues-from across an entire video.

# Task
Analyze the provided detailed video description to exhaustively uncover hidden or complex instances of specific attitudes, emotions, tones, or character traits. You will do this by identifying groups of interconnected video segments that collectively establish how a character feels or thinks about a specific subject. You will demonstrate that the true nature of the sentiment or character trait is only understood by linking the audio of one segment with the visuals of another (or multiple others).

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify All Hidden Sentiment Chains:
Meticulously scan the entire video description to pinpoint all moments where a character's internal state (attitude, emotion, tone) is not immediately obvious from a single moment. Focus on sentiments that are revealed non-linearly.
2. Group Contextually-Linked Segments:
For each sentiment chain you identify, find and list all relevant video segments that, when combined, comprehensively present the character's specific attitude, emotion, or tone. These segments, viewed together, should build a complete picture of the character's state.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must prove that the specific attitude/emotion is only correctly defined when audio elements from one or more segments are combined with visual elements from another one or more segments.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct groups that meet the criteria within the entire video, ensuring that every significant sentiment chain is accounted for.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio elements from one or more segments and visual elements from another one or more segments interact to reveal the complete nature of the sentiment/attitude.]
""".strip()


sentiment_analysis_prompt_2 = """
# Role
You are an expert analyst specializing in multimodal media interpretation and sentiment analysis. Your expertise lies in integrating visual and auditory information from videos (both presented in text format) to understand human psychology and expression.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a highly insightful Q&A pair focused on sentiment analysis.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Internalize the Multimodal Context:
Examine the Designated Segments, referring to the provided connections, fully grasp the complete picture. Identify:
- The Subject: What is the context, object, or event being reacted to?
- The Internal State: What specific attitude, emotion, characteristic, or tone is being displayed?
2. Formulate Targeted Question:
Based on the identified context, draft a single question that asks about what the character's internal state is regarding the subject.
3. Provide Integrated Answer:
In the answer, you must clearly use evidence from the different designated segments-both visual and auditory-and explicitly state how these clues string together to express the full sentiment.

# Constraints:
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question]
A: [Your Answer]
""".strip()


event_sequence_ordering_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Chronological Forensics. Your expertise lies in establishing the definitive sequence of events by synthesizing disparate audio and visual temporal clues from across an entire video.

# Task
Analyze the provided detailed video description to exhaustively establish the correct chronological order of related events. You will do this by identifying and explaining groups of interconnected video segments that cover relevant and easily confused event sequences. For each individual event within a sequence, you must explicitly state whether its occurrence is established through audio evidence, visual evidence, or a combination of both.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Easily Confused Sequences:
Meticulously scan the entire video description to pinpoint sets of related events whose chronological order is ambiguous or easily misinterpreted without careful synthesis.
2. Group All Evidentiary Segments:
For each confusable sequence you identify, find and list all relevant video segments that contain the audio and visual clues needed to solve the timeline.
3. Deconstruct and Prove the Timeline:
For each group, your analysis must first state the correct chronological order of events. Then, you will break down the evidence for each event in that sequence, specifying what happened and the modality of the proof (Audio, Visual, or Audio & Visual). The final sequence should be the only logical conclusion when all the deconstructed evidence is combined.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct groups that meet the criteria within the entire video.
2. Each group you identify must include at least two non-consecutive video segments.
3. The collection of evidence for each group must include at least one audio-based clue and at least one visual-based clue in total.
4. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
Correct Chronological Order:
[A clear, numbered list of the events in their true order.]
Evidentiary Breakdown:
Event 1: [A brief description of what happened.]
Evidence Modality: [Choose one: Audio / Visual / Audio & Visual]
Event 2: [A brief description of what happened.]
Evidence Modality: [Choose one: Audio / Visual / Audio & Visual]
[Add more events as necessary]
Proof of Sequence:
[A concluding statement explaining how assembling these events, identified through their specific modalities, definitively proves the stated chronological order and resolves any initial confusion.]
""".strip()


event_sequence_ordering_prompt_2 = """
# Role
You are an expert analyst specializing in multimodal media interpretation and temporal reasoning. Your expertise lies in articulating complex chronological puzzles by synthesizing pre-identified visual and auditory evidence from a video.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a single, well-formed sequencing question that requires the user to arrange a set of lettered events into their correct chronological order.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Isolate Key Events:
Examine the Designated Segments, referring to the provided connections, identify the distinct, ordered events that form the core of the temporal puzzle.
2. Formulate the Question:
Write a clear and concise question that instructs the user to determine the correct chronological sequence of the lettered events.
3. Present Jumbled, Lettered Events:
Create a lettered list (A, B, C...) of the key events. This list must be presented in a scrambled or neutral order to create the puzzle.
4. Provide the Correct Letter Sequence:
In the answer, state the correct chronological sequence using the letters assigned to the events (e.g. C A B).

# Constraints
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question, e.g., "What is the correct chronological sequence for the following events?]
Events:
A. [Clear and concise description of an event]
B. [Clear and concise description of another event, out of order]
C. [Clear and concise description of the final event, out of order]
...
Correct Sequence: [The correct sequence of letters, e.g., C A B]
""".strip()


summarization_prompt_1 = """
# Role
You are an expert Video Analyst specializing in Topical Cohesion. Your expertise lies in identifying and grouping disparate video segments that, when combined, collectively explain a specific process, describe a multi-part event, or develop a central topic.

# Task
Analyze the provided detailed video description to identify all groups of interconnected segments that collectively cover a single, coherent topic. For each topic, you will group the relevant segments and explain how they build a complete picture through a combination of audio and visual information.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Coherent Topics:
Meticulously scan the entire video description to identify significant topics that are explained or developed across multiple segments. These can be abstract themes (e.g., "a character's guilt") or concrete processes/events (e.g., "the arguments for a new plan").
2. Group All Supporting Segments:
For each distinct topic identified, find and list all relevant video segments that contribute to its explanation or development. These segments, viewed together, should provide a comprehensive understanding of the topic.
3. Analyze the Cross-Modal Development:
For each group, your analysis must explain how the topic is developed through a synthesis of audio and visual elements across the different segments. You must explain how combining audio elements from one or more segments with visual elements from another one or more segments creates a richer understanding of the topic.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct thematic groups that meet the criteria within the entire video.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues and visual evidence from the various segments interact to provide a complete explanation or overview of the topic.]
""".strip()


summarization_prompt_2 = """
# Role
You are an expert Analyst specializing in information synthesis and content summarization. Your expertise lies in articulating a clear overview of a topic, process, or event by weaving together key audio and visual details into a coherent summary.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a single Q&A pair where the question asks for a summary of the topic, and the answer provides a comprehensive, evidence-based summary.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Internalize the Topic:
Examine the Designated Segments, referring to the provided connections, fully grasp the central topic-be it a process, an event, or an explanation-and how it is developed across the clips.
2. Formulate a Summary Question:
Craft a single, open-ended question that asks for a summary or overview of the identified topic.
3. Construct a Synthesized Answer:
In the answer, you must clearly weave together specific auditory and visual evidence from the different segments to state the summary.

# Constraints:
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question]
A: [Your Answer]
""".strip()


causal_reasoning_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Causal Chain Analysis. Your expertise lies in uncovering the root causes of events by synthesizing disparate elements-dialogue, sounds, actions, and visual cues-from across an entire video.

# Task
Analyze the provided detailed video description to exhaustively uncover all hidden or complex causal relationships. You will do this by identifying and explaining groups of interconnected video segments that collectively establish a clear cause-and-effect chain. You will demonstrate that the true reason for a key event or outcome is only understood by linking the audio of one segment with the visuals of another (or multiple others).

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify All Hidden Causal Chains:
Meticulously scan the entire video description to pinpoint all events or outcomes whose root causes are not immediately apparent. Focus on cause-and-effect relationships that are revealed non-linearly or implicitly.
2. Group All Causally-Linked Segments:
For each causal chain you identify, find and list all relevant video segments that, when combined, connect a cause to its effect. These segments, viewed together, should build a complete picture of why a specific outcome occurred.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must prove that the causal link is only correctly and fully understood when audio elements from one or more segments are combined with visual elements from another one or more segments.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct groups that meet the criteria within the entire video, ensuring that every significant causal chain is accounted for.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio elements from one or more segments and visual elements from another one or more segments interact to reveal the complete causal chain.]
""".strip()


causal_reasoning_prompt_2 = """
# Role
You are an expert analyst specializing in multimodal media interpretation and causal reasoning. Your expertise lies in integrating visual and auditory information from videos (both presented in text format) to understand the cause-and-effect relationships between events.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a highly insightful Q&A pair focused on causal reasoning.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Internalize the Causal Chain:
Examine the Designated Segments, referring to the provided connections, fully grasp the cause-and-effect relationship that has been identified. Pinpoint the "cause" and the "effect".
2. Formulate a Precise Causal Question:
Based on the identified chain, craft a single question that asks why the event in the "effect" happened. The question must be unanswerable without referencing the events, dialogue, or visuals in the "cause" segments.
3. Provide Integrated Answers:
In the answer, you must clearly use evidence from the different designated segments-both visual and auditory-and explicitly state how these clues string together to form a complete, logical explanation.

# Constraints:
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question]
A: [Your Answer]
""".strip()


future_prediction_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Predictive Narrative Analysis. Your expertise lies in forecasting future events that are likely to occur after the conclusion of a video by synthesizing disparate audio and visual clues from across the entire narrative.

# Task
Analyze the provided detailed video description to forecast plausible future events. You will do this by identifying and explaining groups of interconnected video segments that collectively provide a strong evidentiary basis for a specific prediction. You will demonstrate that the prediction is not mere speculation, but a logical extrapolation based on the audio and visual clues presented within the video.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify All Predictive Evidence:
Meticulously scan the entire video description to pinpoint all unresolved plot points, stated but unfulfilled intentions, established character behavior patterns, and lingering conflicts.
2. Group All Supporting Segments:
For each plausible future event you predict, find and list all relevant video segments that, when combined, form the logical foundation for that prediction. These segments, viewed together, should build a compelling case for the forecasted outcome.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must prove that the prediction is a well-supported inference derived specifically from a cross-modal synthesis. You must explain how combining audio elements from one or more segments with visual elements from another one or more segments logically points to a specific event that is likely to occur after the video's conclusion.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct, well-supported predictions that can be made from the evidence within the entire video.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues from one or more segments and visual evidence from another one or more segments interact to build a compelling, logical case for the predicted future event.]
""".strip()


future_prediction_prompt_2 = """
# Role
You are an expert analyst specializing in predictive narrative reasoning. Your expertise lies in articulating well-reasoned forecasts about future events by synthesizing pre-identified visual and auditory clues from a video's narrative.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a single, insightful Q&A pair that asks what is likely to happen after the video concludes and provides a detailed, evidence-based justification.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Internalize the Predictive Logic:
Examine the Designated Segments, referring to the provided connections, fully grasp the prediction and the specific clues that support it.
2. Formulate a Predictive Question:
Craft a single, open-ended question that asks for the most likely future event concerning a specific character, object, or situation from the video. The question should probe for a logical extension of the narrative, not a random guess.
3. Provide Integrated Answers:
In the answer, you must clearly cite evidence from the different designated segments-both visual and auditory-and explicitly state how these clues string together to form a complete, logical forecast.

# Constraints:
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question]
A: [Your Answer]
""".strip()


hypothetical_reasoning_prompt_1 = """
# Role
You are an expert Video Content Analyst, specializing in Causal Inference and Counterfactual Simulation. Your expertise lies in constructing "what-if" scenarios by synthesizing interdependent audio and visual clues.

# Task
Analyze the provided detailed video description to construct plausible hypothetical reasoning scenarios. You will do this by identifying specific events or conditions that, if changed (the Hypothesis), would lead to a different, predictable outcome (the Consequence). The logic must require combining audio evidence from one or more segments with visual evidence from another. If the reasoning works using only visual or only audio, it is not a valid output.

# Input
You will be provided with the following information derived from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
SEGMENTS:
{SEGMENTS}

# Instructions
1. Identify Cross-Modal Divergence Points:
Meticulously scan the entire video description to pinpoint moments where the outcome was determined by the interplay of audio and visual factors. Ask: "If this audio/visual condition were different, waht would happen?"
2. Group All Supporting Segments:
For each hypothetical scenario, find and list all relevant video segments. You need segments that establish the original context and segments that provide the evidence for the new outcome.
3. Analyze the Cross-Modal Synthesis:
For each group, your analysis must explain how combining audio elements and visual elements from different segments proves that your hypothetical outcome is grounded in the video's internal logic, not random guessing.

# Constraints
1. Your analysis must be exhaustive. You are required to identify all distinct, well-supported hypothetical scenarios capable of being inferred from the video.
2. Each group you identify must include at least two non-consecutive video segments.
3. If you cannot identify any groups of segments that meet all the above criteria after a thorough analysis, your entire response must be the single word NONE. Do not provide any other text or explanation.

# Output Format
Format your response precisely as follows for each group you identify:
Associated Group [Number]: [A concise title describing the group's central theme]
Relevant Segments:
[Timestamp of the first relevant segment]
[Timestamp of the second relevant segment]
[Add all other relevant timestamps as needed]
Analysis of Connection:
[Detail precisely how audio clues from one or more segments and visual evidence from another one or more segments interact to prove that if the hypothesis occurred, the reasoned consequence would inevitably follow based on the video's logic.]
""".strip()


hypothetical_reasoning_prompt_2 = """
# Role
You are an expert analyst specializing in multimodal counterfactual reasoning. Your expertise lies in articulating well-reasoned "What-if" scenarios where the answer depends entirely on synthesizing visual context with auditory cues.

# Task
Analyze the provided textual description of a video (including a summary, main entities, detailed script, a designated set of segments, and auxiliary analysis) to generate a single, insightful Q&A pair. The question must propose a hypothetical change, and the answer must justify the consequence by explicitly citing both visual and auditory evidence.

# Input
You will be provided with the following information extracted from a video:
Video Summary:
{VIDEO_SUMMARY}
Main Entities:
{MAIN_ENTITIES}
Detailed Script:
{SEGMENTS}
Designated Segments:
{DESIGNATED_SEGMENTS}
Connections
{CONNECTIONS}

# Instructions
1. Internalize the Cross-Modal Logic:
Examine the Designated Segments, referring to the provided connections, understand how the visual state and audio cues interact to create the logical rule for the hypothetical scenario.
2. Formulate Hypothetical Question:
Craft a single, open-ended question that asks "What ... if..." regarding a specific event. The question should target a situation where the answer isn't obvious without considering both sound and sight.
3. Provide Integrated Multimodal Answer:
In the answer, you must clearly cite evidence from the different designated segments-both visual and auditory-and explicitly state how these clues combined necessitate the specific outcome.

# Constraints:
1. Cross-Segment Connection: Q&A pair must be constructed by connecting and synthesizing information from different segments within the Designated Segments. The question and answer cannot be based on a single segment alone.
2. Multimodal Synthesis: Question must depend on both visual information and auditory information. Any answer that could be derived from a single modality (just watching or just listening) is not acceptable.
3. Question must not refer to any 'provided analysis,' 'evidence,' or the process of deduction. It must be a direct question about the events in the video.
4. No Timestamps or Time Ranges: Both the question and the answer must be written in a natural, narrative style. Strictly avoid referring to specific timestamps, time ranges, time codes, or segment IDs in either the question or the answer.

# Output Format
Please adhere strictly to the following format, without any additional preamble or conclusion:
Q: [Your Question]
A: [Your Answer]
""".strip()


API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
TIMEOUT_LIMIT = int(os.environ.get("TIMEOUT_LIMIT", 300))
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", 50))
BASEURL_POOL = os.environ.get("BASEURL_POOL", None).split(",")

qa_num_per_video = int(os.environ.get("QA_NUM", 2))
cross_segment_tasks = ["comparison", "sentiment_analysis", "event_sequence_ordering", "summarization",
                       "causal_reasoning", "future_prediction", "hypothetical_reasoning"]
single_segment_tasks = ["fine_grained_perception", "scene_transformation_detection", "context_understanding"]
prompts = {
    "fine_grained_perception_prompt": fine_grained_perception_prompt,
    "scene_transformation_detection_prompt": scene_transformation_detection_prompt,
    "context_understanding_prompt": context_understanding_prompt,
    "comparison_prompt_1": comparison_prompt_1,
    "comparison_prompt_2": comparison_prompt_2,
    "sentiment_analysis_prompt_1": sentiment_analysis_prompt_1,
    "sentiment_analysis_prompt_2": sentiment_analysis_prompt_2,
    "event_sequence_ordering_prompt_1": event_sequence_ordering_prompt_1,
    "event_sequence_ordering_prompt_2": event_sequence_ordering_prompt_2,
    "summarization_prompt_1": summarization_prompt_1,
    "summarization_prompt_2": summarization_prompt_2,
    "causal_reasoning_prompt_1": causal_reasoning_prompt_1,
    "causal_reasoning_prompt_2": causal_reasoning_prompt_2,
    "future_prediction_prompt_1": future_prediction_prompt_1,
    "future_prediction_prompt_2": future_prediction_prompt_2,
    "hypothetical_reasoning_prompt_1": hypothetical_reasoning_prompt_1,
    "hypothetical_reasoning_prompt_2": hypothetical_reasoning_prompt_2
}


def get_args():
    parser = argparse.ArgumentParser()
    all_tasks = cross_segment_tasks + single_segment_tasks

    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--task", required=True, choices=all_tasks)

    args = parser.parse_args()
    args.script_file = os.path.join(args.root_path, "script.jsonl")
    args.qa_folder = os.path.join(args.root_path, "qa_files")
    return args


def mmss_to_seconds(mmss):
    mm, ss = mmss.split(":")
    return int(mm) * 60 + int(ss)


def get_segments_description(item):
    segments_description = ""
    for seg in item["segments"]:
        segments_description += f"[{seg["start_time"]} - {seg["end_time"]}]\n"

        audio = sorted(seg["transcription"] + seg["non_speech"], key=lambda x: (mmss_to_seconds(x["start_time"]), mmss_to_seconds(x["end_time"])))
        audio_str = ""
        for a in audio:
            if "text" in a:
                audio_str += f"({a["start_time"]}-{a["end_time"]}) [{a["speaker"]}]: {a["text"]}\n"
            if "sound" in a:
                audio_str += f"({a["start_time"]}-{a["end_time"]}) ({a["sound"]})\n"
        if audio_str != "":
            segments_description += f"AUDIO:\n{audio_str}"
                                    
        visual_str = ""
        if len(seg["visual"]) == 1:
            for v in seg["visual"]:
                if "text" in v:
                    visual_str += f"{v["text"].replace("\n\n", "\n")}\n"
        else:
            for v in seg["visual"]:
                if "text" in v:
                    visual_str += f"({v["start_time"]}-{v["end_time"]})\n{v["text"].replace("\n\n", "\n")}\n"
                else:
                    visual_str += f"({v["start_time"]}-{v["end_time"]})\n"
        if visual_str != "":
            segments_description += f"VISUAL:\n{visual_str}"
    return segments_description


async def call_api(client, prompt, timeout):
    def sync_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    
    try:
        response = await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=timeout)
        return response
    except asyncio.TimeoutError:
        return None


async def process_video_cross_1(item, clients, prompt, semaphore, file_lock, file_handle):
    video_id = item["id"]
                            
    async with semaphore:
        main_entites_str = ""
        for entity in item["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        segments_description = get_segments_description(item)
        
        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client,
                                          prompt.format(VIDEO_SUMMARY=item["video_summary"].strip(),
                                                        MAIN_ENTITIES=main_entites_str.strip(),
                                                        SEGMENTS=segments_description.strip()),
                                          timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                item["segment_groups"] = response.text

                if item["segment_groups"] == "NONE":
                    item["qa"] = None
                else:
                    pattern = r"Relevant Segments:\s*(.*?)\s*Analysis of Connection:\s*(.*?)(?=\s*Associated Group|$)"
                    matches = re.findall(pattern, response.text, re.DOTALL)
                    item["qa"] = []
                    for designated_segments, connections in matches:
                        item["qa"].append({"designated_segments": designated_segments.strip(),
                                           "connections": connections.strip()})
                # print(item["qa"])
                
                run_end = time.time()
                tokens_data = response.usage_metadata
                item.setdefault("run_data", {})
                item["run_data"]["segment_groups"] = {
                    "input": tokens_data.prompt_token_count,
                    "output": tokens_data.candidates_token_count,
                    "thinking": tokens_data.thoughts_token_count,
                    "cost_time": run_end - run_start
                }

                async with file_lock:
                    await file_handle.write(json.dumps(item) + "\n")
                    await file_handle.flush()
                print("[SUCESS]", video_id)
                return

            except Exception as e:
                print(f"[Retry]: Error on {video_id} {e}")

        print(f"[FAILED]: {video_id} failed after {len(clients)} retries")
        return


async def cross_segment_task_prompt_1(clients, task, script_data, segment_groups_file):
    data = script_data.copy()
    if os.path.exists(segment_groups_file):
        with open(segment_groups_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item
        
    unprocessed_data = []
    for item in data.values():
        if "qa" in item:
            continue
        unprocessed_data.append(item)
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(segment_groups_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_video_cross_1(item, clients, prompts[f"{task}_prompt_1"], semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
    
    return


async def process_video_cross_2(item, clients, prompt, semaphore, file_lock, file_handle):
    video_id = item["id"]
                            
    async with semaphore:
        main_entites_str = ""
        for entity in item["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        segments_description = get_segments_description(item)
        
        completed_qa_idx = []
        for qa_idx, i in enumerate(item["qa"]):
            if "content" in i and i["content"]:
                completed_qa_idx.append(qa_idx)
        unfinished_qa_idx = [idx for idx in range(len(item["qa"])) if idx not in completed_qa_idx]
        sample_qa_idx = random.sample(unfinished_qa_idx, min(len(unfinished_qa_idx), qa_num_per_video - len(completed_qa_idx)))
        
        for qa_idx in sample_qa_idx:
            i = item["qa"][qa_idx]

            idx = 0
            while idx < len(clients):
                try:
                    run_start = time.time()

                    response = await call_api(clients[idx],
                                            prompt.format(VIDEO_SUMMARY=item["video_summary"].strip(),
                                                            MAIN_ENTITIES=main_entites_str.strip(),
                                                            SEGMENTS=segments_description.strip(),
                                                            DESIGNATED_SEGMENTS=i["designated_segments"].strip(),
                                                            CONNECTIONS=i["connections"].strip()),
                                            timeout=TIMEOUT_LIMIT)
                    if not response:
                        print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                        continue
                    
                    i["content"] = response.text
                    # print(i)
                    
                    run_end = time.time()
                    tokens_data = response.usage_metadata
                    item.setdefault("run_data", {})
                    item["run_data"].setdefault("qa", [])
                    item["run_data"]["qa"].append({
                        "input": tokens_data.prompt_token_count,
                        "output": tokens_data.candidates_token_count,
                        "thinking": tokens_data.thoughts_token_count,
                        "cost_time": run_end - run_start
                    })

                    async with file_lock:
                        await file_handle.write(json.dumps(item) + "\n")
                        await file_handle.flush()
                    print("[SUCESS]", video_id, f"Group {qa_idx}")
                    break

                except Exception as e:
                    print(f"[Retry]: Error on {video_id} Group {qa_idx} {e}")
                    idx += 1

            if idx == len(clients):
                print(f"[FAILED]: {video_id} Group {qa_idx} failed after {len(clients)} retries")


async def cross_segment_task_prompt_2(clients, task, segment_groups_file, qa_file):
    data = {}
    with open(segment_groups_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            data[item["id"]] = item
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item
    
    unprocessed_data = []
    for item in data.values():
        if not item["qa"]:
            continue

        generated_count = 0
        for i in item["qa"]:
            if "content" in i and i["content"]:
                generated_count += 1
        if generated_count >= qa_num_per_video or generated_count == len(item["qa"]):
            continue
        unprocessed_data.append(item)
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(qa_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_video_cross_2(item, clients, prompts[f"{task}_prompt_2"], semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
    
    return


async def process_video_single(item, clients, prompt, semaphore, file_lock, file_handle):
    video_id = item["id"]

    async with semaphore:
        main_entites_str = ""
        for entity in item["main_entities"]:
            main_entites_str += f"- {entity["entity"]}: {entity["description"]}\n"

        segments_description = get_segments_description(item)
        
        for client in clients:
            try:
                run_start = time.time()

                response = await call_api(client,
                                          prompt.format(VIDEO_SUMMARY=item["video_summary"].strip(),
                                                        MAIN_ENTITIES=main_entites_str.strip(),
                                                        SEGMENTS=segments_description.strip()),
                                          timeout=TIMEOUT_LIMIT)
                if not response:
                    print("[TIMEOUT] API call exceeded timeout:", TIMEOUT_LIMIT)
                    continue

                item["qa"] = response.text
                # print(item["qa"])
                
                run_end = time.time()
                tokens_data = response.usage_metadata
                item.setdefault("run_data", {})
                item["run_data"]["qa"] = {
                    "input": tokens_data.prompt_token_count,
                    "output": tokens_data.candidates_token_count,
                    "thinking": tokens_data.thoughts_token_count,
                    "cost_time": run_end - run_start
                }

                async with file_lock:
                    await file_handle.write(json.dumps(item) + "\n")
                    await file_handle.flush()
                print("[SUCESS]", video_id)
                return

            except Exception as e:
                print(f"[Retry]: Error on {video_id} {e}")

        print(f"[FAILED]: {video_id} failed after {len(clients)} retries")
        return


async def single_segment_task_prompt(clients, task, script_data, qa_file):
    data = script_data.copy()
    if os.path.exists(qa_file):
        with open(qa_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                data[item["id"]] = item
    
    unprocessed_data = []
    for item in data.values():
        if "qa" in item:
            continue
        unprocessed_data.append(item)
    
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    async with aiofiles.open(qa_file, "a", encoding="utf-8") as f:
        tasks = []
        for item in unprocessed_data:
            tasks.append(asyncio.create_task(process_video_single(item, clients, prompts[f"{task}_prompt"], semaphore, file_lock, f)))

        if tasks:
            for f_task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Videos"):
                await f_task
    
    return


async def main():
    args = get_args()
    task = args.task
    os.makedirs(args.qa_folder, exist_ok=True)
    if BASEURL_POOL:
        clients = [genai.Client(
            api_key=API_KEY,
            http_options={
                "base_url": base_url
            }
        ) for base_url in BASEURL_POOL]
    else:
        clients = [genai.Client(api_key=API_KEY)]

    script_data = {}
    with open(args.script_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            item = json.loads(line)
            script_data[item["id"]] = item

    if task in cross_segment_tasks:
        print(f"{task} begin")
        segment_groups_file = os.path.join(args.qa_folder, f"{task}_segment_groups.jsonl")
        await cross_segment_task_prompt_1(clients, task, script_data=script_data, segment_groups_file=segment_groups_file)

        qa_file = os.path.join(args.qa_folder, f"{task}_qa.jsonl")
        await cross_segment_task_prompt_2(clients, task, segment_groups_file=segment_groups_file, qa_file=qa_file)
        print(f"{task} completed")

    if task in single_segment_tasks:
        print(f"{task} begin")
        qa_file = os.path.join(args.qa_folder, f"{task}_qa.jsonl")
        await single_segment_task_prompt(clients, task, script_data=script_data, qa_file=qa_file)
        print(f"{task} completed")

    os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())
