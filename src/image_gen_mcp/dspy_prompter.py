#!/usr/bin/env python3
"""
DSPy-based Contextual Image Prompt Generator
=============================================

Generates meaningful, context-aware prompts for the Pixel Corgi based on
actual work being performed. Uses DSPy for intelligent prompt generation
that reflects the visual context of the current activity.

Example activities and their visual representations:
- Security scan → Corgi as detective with magnifying glass
- Code review → Corgi with glasses reviewing scrolls
- Git commit → Corgi carrying a package/delivery
- Error debugging → Corgi with confused expression, bugs around
- Research → Corgi in library with books
- Building/compiling → Corgi as construction worker
"""

import dspy
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

# Configure DSPy to use a remote LLM (Ollama on cluster node)
# Avoid local CPU inference - set OLLAMA_HOST to your GPU node
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class WorkContext:
    """Context about the current work activity."""
    event_type: str  # success, error, start, end, tool_use, etc.
    tool_name: Optional[str] = None  # Bash, Edit, Read, etc.
    tool_args: Optional[Dict] = None  # Arguments passed to tool
    task_description: Optional[str] = None  # What the task is about
    error_message: Optional[str] = None  # If there was an error
    file_types: Optional[List[str]] = None  # .py, .rs, .ts, etc.
    domain: Optional[str] = None  # security, web, database, etc.
    mood: str = "neutral"  # excited, focused, confused, tired, celebrating


class ContextToVisualMetaphor(dspy.Signature):
    """Convert work context into a visual metaphor for the Pixel Corgi.

    Think about what visual elements would represent this activity:
    - What props or accessories would the corgi have?
    - What environment/background fits the activity?
    - What pose or expression matches the mood?
    - What small details add context (floating icons, effects)?
    """

    event_type: str = dspy.InputField(desc="Type of event: success, error, thinking, coding, etc.")
    tool_name: str = dspy.InputField(desc="Tool being used: Bash, Edit, Read, Grep, etc.")
    task_summary: str = dspy.InputField(desc="Brief summary of what work is being done")
    domain: str = dspy.InputField(desc="Domain: security, web, database, devops, research, etc.")
    mood: str = dspy.InputField(desc="Current mood: excited, focused, confused, celebrating, tired")

    visual_props: str = dspy.OutputField(desc="Props/accessories for the corgi (e.g., 'tiny hardhat, wrench')")
    environment: str = dspy.OutputField(desc="Background/setting (e.g., 'construction site with code blocks')")
    pose_expression: str = dspy.OutputField(desc="Pose and facial expression (e.g., 'determined stance, tongue out')")
    visual_effects: str = dspy.OutputField(desc="Floating elements/effects (e.g., 'sparkles, binary numbers')")


class GeneratePixelCorgiPrompt(dspy.Signature):
    """Generate a complete NES-style pixel art prompt for the Pixel Corgi.

    The prompt should be optimized for image generation models like Flux.
    Include specific pixel art styling keywords and keep it concise but descriptive.
    """

    visual_props: str = dspy.InputField(desc="Props and accessories for the corgi")
    environment: str = dspy.InputField(desc="Background setting")
    pose_expression: str = dspy.InputField(desc="Pose and expression")
    visual_effects: str = dspy.InputField(desc="Floating elements and effects")

    image_prompt: str = dspy.OutputField(desc="Complete image generation prompt, 50-100 words")


class PixelCorgiPrompter(dspy.Module):
    """DSPy module for generating contextual Pixel Corgi prompts."""

    def __init__(self):
        super().__init__()
        self.context_to_visual = dspy.ChainOfThought(ContextToVisualMetaphor)
        self.generate_prompt = dspy.Predict(GeneratePixelCorgiPrompt)

        # Fallback visual mappings - symbolic fantasy scenes with Pixel leading subagent companions
        self.fallback_mappings = {
            "security": {
                "props": "knight armor, glowing shield",
                "environment": "castle watchtower at dusk, protective barrier dome",
                "pose": "vigilant guardian stance with hawk scout companions",
                "effects": "scanning beams, shield runes glowing, sentinel birds circling"
            },
            "coding": {
                "props": "wizard robe, enchanted quill",
                "environment": "arcane scriptorium tower with floating ancient tomes",
                "pose": "Pixel scribe commanding sprite helpers writing magical runes",
                "effects": "glowing ink trails, runic symbols materializing, quill familiars"
            },
            "git": {
                "props": "postmaster cap, magical satchel",
                "environment": "sky roads with branch pathways, courier station in clouds",
                "pose": "Pixel leading messenger bird fleet delivering glowing parcels",
                "effects": "flying letters, branch-shaped aurora trails, delivery sparkles"
            },
            "database": {
                "props": "librarian robes, crystal orb",
                "environment": "infinite crystal archive dimension with data obelisks",
                "pose": "Pixel curator with mouse archivists organizing glowing records",
                "effects": "floating data crystals, sorting beams, organized constellations"
            },
            "web": {
                "props": "artist beret, magic paintbrush",
                "environment": "floating canvas realm with interface gardens",
                "pose": "Pixel painter with butterfly assistants creating portal windows",
                "effects": "brushstroke trails becoming interfaces, color magic, design sprites"
            },
            "devops": {
                "props": "architect helmet, blueprint scroll",
                "environment": "construction realm with floating building blocks",
                "pose": "Pixel architect directing beaver and ant builder crews",
                "effects": "crane golems, assembling magical structures, gear constellations"
            },
            "research": {
                "props": "explorer hat, discovery compass",
                "environment": "infinite library dimension with knowledge nebulas",
                "pose": "Pixel explorer with wise owl companions discovering ancient secrets",
                "effects": "books opening with light, knowledge orbs floating, eureka sparkles"
            },
            "testing": {
                "props": "alchemist goggles, potion vials",
                "environment": "laboratory tower with experiment chambers",
                "pose": "Pixel scientist with mouse lab assistants running magical tests",
                "effects": "bubbling potions, checkmark crystals forming, verification magic"
            },
            "error": {
                "props": "detective cloak, puzzle piece",
                "environment": "foggy maze with mysterious red glowing fragments",
                "pose": "Pixel and owl companion studying scattered puzzle pieces",
                "effects": "question mark wisps, mystery fog, clue trails appearing"
            },
            "success": {
                "props": "champion cape, golden chalice",
                "environment": "mountain peak at sunrise with treasure revealed",
                "pose": "Pixel celebrating with fairy companion squad, victory leap",
                "effects": "golden light rays, confetti rain, sparkle explosions, triumph fanfare"
            }
        }

    def _detect_domain(self, context: WorkContext) -> str:
        """Detect the work domain from context."""
        tool = (context.tool_name or "").lower()
        task = (context.task_description or "").lower()

        # Security indicators
        if any(x in task for x in ["security", "scan", "vuln", "audit", "pentest", "exploit"]):
            return "security"
        if any(x in tool for x in ["nuclei", "nmap", "security"]):
            return "security"

        # Git indicators
        if any(x in task for x in ["commit", "push", "pull", "merge", "branch", "git"]):
            return "git"
        if "git" in tool:
            return "git"

        # Database indicators
        if any(x in task for x in ["database", "sql", "query", "migration", "schema"]):
            return "database"
        if any(x in tool for x in ["sqlite", "postgres", "mysql", "redis"]):
            return "database"

        # Web indicators
        if any(x in task for x in ["frontend", "css", "html", "react", "vue", "ui", "design"]):
            return "web"

        # DevOps indicators
        if any(x in task for x in ["docker", "deploy", "kubernetes", "ci/cd", "pipeline", "container"]):
            return "devops"

        # Research indicators
        if any(x in task for x in ["research", "paper", "arxiv", "study", "investigate", "analyze"]):
            return "research"

        # Testing indicators
        if any(x in task for x in ["test", "pytest", "jest", "spec", "coverage"]):
            return "testing"

        # Error state
        if context.event_type == "error" or context.error_message:
            return "error"

        # Success state
        if context.event_type == "success":
            return "success"

        # Default to coding
        return "coding"

    def _get_fallback_prompt(self, context: WorkContext) -> str:
        """Generate prompt using fallback mappings when LLM unavailable."""
        domain = context.domain or self._detect_domain(context)
        mapping = self.fallback_mappings.get(domain, self.fallback_mappings["coding"])

        base = "8-bit NES pixel art, cute tricolor corgi Pixel leading team of companions, clean pixels, retro game fantasy scene"

        prompt = f"{base}, {mapping['props']}, {mapping['environment']}, {mapping['pose']}, {mapping['effects']}"

        # Add mood modifiers - fantasy themed
        if context.mood == "excited":
            prompt += ", magical energy aura, companions celebrating, adventure spirit"
        elif context.mood == "tired":
            prompt += ", campfire rest scene, sleepy companions curled up, moonlight"
        elif context.mood == "confused":
            prompt += ", mysterious fog, companions investigating, question wisps"
        elif context.mood == "celebrating":
            prompt += ", victory fireworks, companions cheering, golden triumph"

        return prompt

    def forward(self, context: WorkContext) -> str:
        """Generate a contextual prompt for the Pixel Corgi."""

        # Detect domain if not provided
        domain = context.domain or self._detect_domain(context)

        # Create task summary
        task_summary = context.task_description or f"{context.event_type} event"
        if context.tool_name:
            task_summary = f"Using {context.tool_name}: {task_summary}"

        try:
            # Try to use DSPy with LLM
            visual = self.context_to_visual(
                event_type=context.event_type,
                tool_name=context.tool_name or "general",
                task_summary=task_summary,
                domain=domain,
                mood=context.mood
            )

            result = self.generate_prompt(
                visual_props=visual.visual_props,
                environment=visual.environment,
                pose_expression=visual.pose_expression,
                visual_effects=visual.visual_effects
            )

            # Ensure NES pixel art styling
            prompt = result.image_prompt
            if "pixel" not in prompt.lower():
                prompt = f"8-bit NES pixel art sprite, {prompt}"
            if "corgi" not in prompt.lower():
                prompt = prompt.replace("pixel art sprite", "pixel art sprite cute corgi Pixel")

            return prompt

        except Exception as e:
            # Fallback to rule-based generation
            return self._get_fallback_prompt(context)


def configure_dspy(model: str = "llama3.2:3b"):
    """Configure DSPy with Ollama backend on cluster node."""
    try:
        lm = dspy.LM(
            model=f"ollama_chat/{model}",
            api_base=OLLAMA_HOST,
            api_key="",  # Ollama doesn't need API key
            temperature=0.7,
            max_tokens=200
        )
        dspy.configure(lm=lm)
        return True
    except Exception as e:
        print(f"Failed to configure DSPy: {e}")
        return False


def generate_contextual_prompt(
    event_type: str,
    tool_name: Optional[str] = None,
    task_description: Optional[str] = None,
    error_message: Optional[str] = None,
    mood: str = "neutral",
    use_llm: bool = True
) -> str:
    """
    Generate a contextual Pixel Corgi prompt.

    Args:
        event_type: Type of event (success, error, start, end, tool_use)
        tool_name: Name of the tool being used
        task_description: Description of the current task
        error_message: Error message if applicable
        mood: Current mood (excited, focused, confused, tired, celebrating)
        use_llm: Whether to use LLM (requires cluster node)

    Returns:
        Image generation prompt for Pixel Corgi
    """
    context = WorkContext(
        event_type=event_type,
        tool_name=tool_name,
        task_description=task_description,
        error_message=error_message,
        mood=mood
    )

    prompter = PixelCorgiPrompter()

    if use_llm:
        configured = configure_dspy()
        if not configured:
            use_llm = False

    if use_llm:
        return prompter.forward(context)
    else:
        return prompter._get_fallback_prompt(context)


# CLI interface for shell script integration
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate contextual Pixel Corgi prompts")
    parser.add_argument("--event", default="idle", help="Event type")
    parser.add_argument("--tool", default=None, help="Tool name")
    parser.add_argument("--task", default=None, help="Task description")
    parser.add_argument("--error", default=None, help="Error message")
    parser.add_argument("--mood", default="neutral", help="Mood")
    parser.add_argument("--no-llm", action="store_true", help="Use fallback only (no LLM)")

    args = parser.parse_args()

    prompt = generate_contextual_prompt(
        event_type=args.event,
        tool_name=args.tool,
        task_description=args.task,
        error_message=args.error,
        mood=args.mood,
        use_llm=not args.no_llm
    )

    print(prompt)
