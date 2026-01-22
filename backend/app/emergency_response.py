from __future__ import annotations

import os
from datetime import datetime
from typing import Optional
import google.generativeai as genai
from .models import RiskLevel, Alert


class EmergencyResponseEngine:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_available = False
        self.model = None
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
                self.gemini_available = True
            except Exception:
                pass
    
    def generate_recommendation(
        self,
        alert: Alert,
        area_affected: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        escalation_trend: Optional[str] = None
    ) -> dict:
        """
        Generate emergency response recommendation based on alert data
        
        Args:
            alert: Alert object with risk information
            area_affected: Description of affected area (e.g., "Main entrance", "Food court")
            duration_seconds: How long the risk has persisted
            escalation_trend: Whether risk is "increasing", "stable", or "decreasing"
        
        Returns:
            Dictionary with recommendation data
        """
        if alert.risk_level not in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
            return {
                "time": alert.created_at.strftime('%H:%M:%S'),
                "risk_level": alert.risk_level.value,
                "unit": None,
                "action": None,
                "urgency": None,
                "reasoning": ["Recommendation only generated for MEDIUM and HIGH-risk events"]
            }
        
        prompt = self._build_prompt(alert, area_affected, duration_seconds, escalation_trend)
        
        if not self.gemini_available:
            return self._fallback_recommendation(alert, area_affected, duration_seconds, escalation_trend)
        
        try:
            response = self.model.generate_content(prompt)
            recommendation = self._parse_response(response.text, alert)
            
            return recommendation
            
        except Exception as e:
            return self._fallback_recommendation(alert, area_affected, duration_seconds, escalation_trend)
    
    def _build_prompt(
        self,
        alert: Alert,
        area_affected: Optional[str],
        duration_seconds: Optional[float],
        escalation_trend: Optional[str]
    ) -> str:
        """Build the prompt for Gemini AI"""
        
        duration_text = ""
        if duration_seconds:
            minutes = int(duration_seconds / 60)
            seconds = int(duration_seconds % 60)
            duration_text = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
        
        prompt = f"""You are an emergency response advisor for crowd safety incidents.

Analyze this {alert.risk_level.value}-risk crowd safety incident and provide a specific, actionable response recommendation.

INCIDENT DATA:
- Time: {alert.created_at.strftime('%H:%M:%S')}
- Location: {alert.location}
- Risk Level: {alert.risk_level.value}
- Risk Score: {alert.risk_score:.4f}
- Confidence: {alert.confidence:.2%}
- Primary Cause: {alert.primary_cause}
- Supporting Factors: {', '.join(alert.supporting_factors or [])}
- Explanation: {alert.explanation}
"""
        
        if area_affected:
            prompt += f"- Area Affected: {area_affected}\n"
        if duration_text:
            prompt += f"- Duration: {duration_text}\n"
        if escalation_trend:
            prompt += f"- Escalation Trend: {escalation_trend}\n"
        
        prompt += """
Provide your recommendation in EXACTLY this format (no additional text):

Unit: [Type of response unit needed]
Action: [Primary action to take]
Urgency: [Immediate/High/Medium]
Reasoning:
- [First reasoning point]
- [Second reasoning point]

GUIDELINES:
- Unit types: Crowd Control Police, Tactical Response Team, Emergency Medical Services, Fire Safety Team, Traffic Control Unit
- Action must be specific and actionable
- Urgency levels:
  * "Immediate" for HIGH-risk life-threatening situations requiring instant response
  * "High" for HIGH-risk events with potential for rapid escalation
  * "Medium" for MEDIUM-risk events requiring prompt but measured response
- Reasoning must reference specific data points from the incident
- Keep responses concise and professional
- For MEDIUM risk events, focus on monitoring and preventive measures
- For HIGH risk events, focus on immediate intervention and crowd control
"""
        return prompt
    
    def _parse_response(self, response_text: str, alert: Alert) -> dict:
        lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
        
        recommendation = {
            "time": alert.created_at.strftime('%H:%M:%S'),
            "risk_level": alert.risk_level.value,
            "unit": None,
            "action": None,
            "urgency": None,
            "reasoning": []
        }
        
        current_section = None
        
        for line in lines:
            if line.startswith("Unit:"):
                recommendation["unit"] = line.replace("Unit:", "").strip()
            elif line.startswith("Action:"):
                recommendation["action"] = line.replace("Action:", "").strip()
            elif line.startswith("Urgency:"):
                recommendation["urgency"] = line.replace("Urgency:", "").strip()
            elif line.startswith("Reasoning:"):
                current_section = "reasoning"
            elif current_section == "reasoning" and line.startswith("-"):
                recommendation["reasoning"].append(line[1:].strip())
        
        if not all([recommendation["unit"], recommendation["action"], recommendation["urgency"]]):
            return self._fallback_recommendation(alert, None, None, None)
        
        return recommendation
    
    def _fallback_recommendation(
        self,
        alert: Alert,
        area_affected: Optional[str],
        duration_seconds: Optional[float],
        escalation_trend: Optional[str]
    ) -> dict:

        primary = (alert.primary_cause or "").lower()
        factors = [f.lower() for f in (alert.supporting_factors or [])]
        
        if alert.risk_level == RiskLevel.HIGH:
            unit = "Crowd Control Police"
            action = "Restrict entry points, manage flow, establish control zones"
            urgency = "Immediate"
        else:  
            unit = "Crowd Control Police"
            action = "Monitor situation closely, prepare for potential intervention"
            urgency = "Medium"
        
        reasoning = []
        
        if "density" in primary or "overcrowding" in primary:
            unit = "Crowd Control Police"
            if alert.risk_level == RiskLevel.HIGH:
                action = "Restrict entry points, manage crowd flow, establish buffer zones"
                urgency = "Immediate"
                reasoning.append("High crowd density requires immediate intervention")
            else:
                action = "Monitor crowd density, prepare flow control measures"
                urgency = "Medium"
                reasoning.append("Elevated crowd density detected - monitoring required")
            
        elif "panic" in primary or "stampede" in primary:
            unit = "Tactical Response Team"
            action = "Create evacuation corridors, establish safe zones, prevent crushing"
            urgency = "Immediate"
            reasoning.append("Panic or stampede risk - life-threatening situation")
            
        elif "fire" in primary or "smoke" in primary:
            unit = "Fire Safety Team"
            action = "Immediate evacuation, secure fire exits, crowd dispersal"
            urgency = "Immediate"
            reasoning.append("Fire hazard detected - immediate evacuation required")
            
        elif "medical" in primary or "injury" in primary:
            unit = "Emergency Medical Services"
            if alert.risk_level == RiskLevel.HIGH:
                action = "Clear path for medical access, crowd separation, triage area setup"
                urgency = "Immediate"
            else:
                action = "Position medical team on standby, monitor situation"
                urgency = "High"
            reasoning.append("Medical situation requires prompt response")
            
        elif "traffic" in primary or "vehicle" in primary:
            unit = "Traffic Control Unit"
            action = "Redirect traffic, establish pedestrian barriers, secure area"
            urgency = "High" if alert.risk_level == RiskLevel.HIGH else "Medium"
            reasoning.append("Traffic-related crowd safety issue")
        
        elif "anomaly" in primary or "persistence" in primary:
            if alert.risk_level == RiskLevel.HIGH:
                action = "Deploy surveillance team, assess escalation potential, prepare response"
                urgency = "High"
            else:
                action = "Increase monitoring, document patterns, alert response teams"
                urgency = "Medium"
            reasoning.append("Sustained anomaly detected requiring attention")
        
        # Add reasoning based on supporting factors
        if "directional chaos" in ' '.join(factors):
            reasoning.append("Directional chaos indicates potential loss of crowd control")
            
        if duration_seconds and duration_seconds > 30:
            reasoning.append(f"Risk persistence over {int(duration_seconds)}s indicates sustained threat")
            
        if escalation_trend == "increasing":
            reasoning.append("Escalating risk trend requires immediate intervention")
            urgency = "Immediate"
        
        if len(reasoning) < 2:
            if alert.risk_level == RiskLevel.HIGH:
                reasoning.append(f"Risk score {alert.risk_score:.4f} exceeds HIGH threshold")
            else:
                reasoning.append(f"Risk score {alert.risk_score:.4f} indicates MEDIUM-level concern")
        
        return {
            "time": alert.created_at.strftime('%H:%M:%S'),
            "risk_level": alert.risk_level.value,
            "unit": unit,
            "action": action,
            "urgency": urgency,
            "reasoning": reasoning
        }


_engine: Optional[EmergencyResponseEngine] = None


def get_emergency_response_engine() -> EmergencyResponseEngine:
    global _engine
    if _engine is None:
        _engine = EmergencyResponseEngine()
    return _engine
