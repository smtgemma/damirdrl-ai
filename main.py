import os
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Literal
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from google import genai
from google.genai import types
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create journals directory if it doesn't exist
JOURNALS_DIR = Path("journals")
JOURNALS_DIR.mkdir(exist_ok=True)

# ============================================================================
# STEP 1: Define Data Models (ALL FIELDS REQUIRED)
# ============================================================================

class CountryData(BaseModel):
    """Information about one country you're visiting - ALL FIELDS REQUIRED"""
    country_name: str = Field(..., description="Country name", example="Thailand")
    rural_stay: bool = Field(..., description="Will you stay in rural/forest areas?")
    close_contact_local_pop: bool = Field(..., description="Close contact with local population?")
    staying_with_family: bool = Field(..., description="Staying with local family?")
    close_contact_animals: bool = Field(..., description="Close contact with animals?")
    risky_activities: bool = Field(..., description="Tattoos, surgery, healthcare work?")
    departure_date: date = Field(..., description="Trip departure date", example="2025-11-15")
    return_date: date = Field(..., description="Trip return date", example="2025-12-15")
    
    @validator('return_date')
    def validate_return_date(cls, v, values):
        """Ensure return date is after departure date"""
        if 'departure_date' in values and v <= values['departure_date']:
            raise ValueError('Return date must be after departure date')
        return v
    
    @property
    def duration_of_stay(self) -> int:
        """Calculate duration in days"""
        return (self.return_date - self.departure_date).days


class TravelerInfo(BaseModel):
    """Basic traveler information"""
    age: int = Field(..., ge=0, le=120, description="Traveler age", example=35)


class TravelRequest(BaseModel):
    """The complete request sent to the API"""
    booking_id: int = Field(..., description="Booking ID", example=28066531)
    token: str = Field(..., description="Authentication token", example="wefiushfsidfhsiufhsifnsdfhnsiud")
    traveler_info: TravelerInfo
    countries: List[CountryData] = Field(..., min_items=1, max_items=10)


# ============================================================================
# STRUCTURED OUTPUT SCHEMA FOR AI
# ============================================================================

class VaccineDose(BaseModel):
    """Single vaccine dose information"""
    dose_number: int = Field(..., description="Dose number (1, 2, 3, etc.)")
    timing_description: str = Field(..., description="When to take this dose in plain language")
    days_from_today: Optional[int] = Field(None, description="Approximate days from today (if calculable)")


class VaccineScheduleOption(BaseModel):
    """One scheduling option for a vaccine (e.g., standard vs accelerated)"""
    option_name: str = Field(..., description="Name of this schedule option (e.g., 'Standard Schedule', 'Accelerated Schedule', 'Oral Capsules')")
    doses: List[VaccineDose] = Field(..., description="List of doses in this schedule")
    administration_notes: Optional[str] = Field(None, description="Additional notes about administration")


class ConsultationVisit(BaseModel):
    """One consultation visit with multiple vaccines"""
    visit_number: int = Field(..., description="Consultation number (1, 2, 3, etc.)")
    timing_description: str = Field(..., description="When this visit should occur")
    days_from_today: Optional[int] = Field(None, description="Days from today")
    vaccines_to_administer: List[str] = Field(..., description="List of vaccines given in this visit")
    administration_notes: Optional[str] = Field(None, description="Notes for this visit")
    overlap_warning: Optional[str] = Field(None, description="Warning if visit falls during travel")


class VaccineProtection(BaseModel):
    """Protection timeline for one vaccine"""
    vaccine_name: str = Field(..., description="Name of the vaccine")
    protection_onset: str = Field(..., description="When protection begins")
    full_protection: str = Field(..., description="When full protection is achieved")
    immunity_duration: str = Field(..., description="How long immunity lasts")
    booster_info: Optional[str] = Field(None, description="Information about booster requirements")


class MalariaProtocol(BaseModel):
    """Malaria prevention information"""
    is_required: bool = Field(..., description="Whether malaria prophylaxis is required")
    medication_name: Optional[str] = Field(None, description="Name and dosage of medication")
    start_timing: Optional[str] = Field(None, description="When to start taking medication")
    stop_timing: Optional[str] = Field(None, description="When to stop taking medication")
    administration_instructions: Optional[str] = Field(None, description="How to take the medication")
    side_effects: Optional[str] = Field(None, description="Common side effects")
    additional_protection: Optional[str] = Field(None, description="Additional protective measures")


class TravelSummary(BaseModel):
    """Summary of travel information"""
    destinations: str = Field(..., description="List of destination countries")
    total_trip_duration_days: int = Field(..., description="Total trip duration in days")
    days_until_departure: int = Field(..., description="Days until departure")
    rural_or_forest_areas: bool = Field(..., description="Whether visiting rural/forest areas")
    contact_with_locals: bool = Field(..., description="Whether close contact with locals")
    staying_with_locals: bool = Field(..., description="Whether staying with local families")
    animal_contact: bool = Field(..., description="Whether potential animal contact")
    risky_activities: bool = Field(..., description="Whether risky activities planned")
    departure_date: str = Field(..., description="Departure date")
    final_return_date: str = Field(..., description="Final return date")
    traveler_age: int = Field(..., description="Traveler's age")


class StructuredHealthPlan(BaseModel):
    """Structured output schema for AI-generated health plan"""
    recommended_vaccines: List[str] = Field(..., description="List of recommended vaccine names")
    malaria_prevention_required: bool = Field(..., description="Whether malaria prevention is needed")
    travel_summary: TravelSummary = Field(..., description="Summary of travel information")
    vaccination_schedules: List[ConsultationVisit] = Field(..., description="Detailed vaccination schedules")
    vaccine_protections: List[VaccineProtection] = Field(..., description="Protection timelines for each vaccine")
    malaria_protocol: MalariaProtocol = Field(..., description="Malaria prevention protocol")


# class HealthPlanResponse(BaseModel):
#     """Response with health plan"""
#     status: str
#     health_plan: str
#     structured_data: StructuredHealthPlan
#     journal_file: str
#     journal_download_path: str
#     countries_analyzed: int
#     traveler_age: int
#     sources_used: Optional[List[str]] = None

class HealthPlanResponse(BaseModel):
    """Response with health plan"""
    health_plan: str


# ============================================================================
# STEP 2: Setup Gemini API with Grounding
# ============================================================================

def setup_gemini_client():
    """Initialize the Gemini API client"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY not found in .env file"
        )
    
    try:
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client connected successfully")
        return client
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Gemini: {str(e)}"
        )


# Create global client
gemini_client = setup_gemini_client()


# ============================================================================
# HELPER: Clean JSON Response
# ============================================================================

def clean_json_response(text: str) -> str:
    """
    Remove markdown code blocks and other artifacts from JSON response.
    Handles various formats:
    - ```json ... ```
    - ``` ... ```
    - ```{ ... }```
    """
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove markdown code blocks with various patterns
    # Pattern 1: ```json\n...\n```
    text = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
    # Pattern 2: ```\n...\n```
    text = re.sub(r'^```\s*\n', '', text, flags=re.MULTILINE)
    # Pattern 3: ```{ at start
    text = re.sub(r'^```\s*\{', '{', text)
    # Pattern 4: ``` at end
    text = re.sub(r'\}\s*```$', '}', text)
    # Pattern 5: standalone ``` at start or end
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    # Strip again after removal
    text = text.strip()
    
    return text


# ============================================================================
# STEP 3: Create the Unified Prompt with Structured Output
# ============================================================================

def create_unified_prompt(data: TravelRequest) -> str:
    """
    Create a comprehensive prompt that requests structured output from AI.
    This eliminates bias from examples and ensures consistent format.
    """
    
    # Get current date
    current_date = date.today()

    # Calculate travel day numbers relative to today
    travel_day_starts = (data.countries[0].departure_date - current_date).days  # Day 0 = today
    travel_day_ends = (data.countries[-1].return_date - current_date).days
    
    # Initialize aggregate risk factors
    any_rural_stay = False
    any_close_contact_local_pop = False
    any_staying_with_family = False
    any_close_contact_animals = False
    any_risky_activities = False
    
    # Build detailed country information
    countries_details = ""
    total_trip_days = 0
    
    for idx, country in enumerate(data.countries, 1):
        trip_duration = country.duration_of_stay
        total_trip_days += trip_duration
        
        # Aggregate risk factors
        any_rural_stay = any_rural_stay or country.rural_stay
        any_close_contact_local_pop = any_close_contact_local_pop or country.close_contact_local_pop
        any_staying_with_family = any_staying_with_family or country.staying_with_family
        any_close_contact_animals = any_close_contact_animals or country.close_contact_animals
        any_risky_activities = any_risky_activities or country.risky_activities
        
        countries_details += f"""
COUNTRY {idx}: {country.country_name}
Trip Duration: {trip_duration} days (from {country.departure_date} to {country.return_date})
Risk Factors:
- Rural/forest areas: {'Yes' if country.rural_stay else 'No'}
- Close contact with locals: {'Yes' if country.close_contact_local_pop else 'No'}
- Staying with local family: {'Yes' if country.staying_with_family else 'No'}
- Animal contact: {'Yes' if country.close_contact_animals else 'No'}
- Risky activities: {'Yes' if country.risky_activities else 'No'}
"""
    
    countries_list = ", ".join([c.country_name for c in data.countries])
    current_date_str = current_date.strftime("%d %b %Y")
    departure_date_str = data.countries[0].departure_date.strftime("%d %b %Y")
    last_return_date_str = data.countries[-1].return_date.strftime("%d %b %Y")
    
    # Calculate days until departure
    days_until_departure = (data.countries[0].departure_date - current_date).days
    days_until_return = (data.countries[-1].return_date - current_date).days
    
    # Create aggregate risk summary
    aggregate_risks = f"""
AGGREGATE RISK FACTORS ACROSS ALL DESTINATIONS:
- Rural/forest areas: {'Yes' if any_rural_stay else 'No'}
- Close contact with locals: {'Yes' if any_close_contact_local_pop else 'No'}
- Staying with local family: {'Yes' if any_staying_with_family else 'No'}
- Animal contact: {'Yes' if any_close_contact_animals else 'No'}
- Risky activities: {'Yes' if any_risky_activities else 'No'}

IMPORTANT: If ANY of the above risks are "Yes", provide recommendations for that risk factor.
"""
    
    prompt = f"""You are a travel medicine specialist. Use web search to find current recommendations from CDC (wwwnc.cdc.gov/travel) and SSI Denmark (rejse.ssi.dk) for the traveler's specific destinations and risk profile.

TRAVELER PROFILE:
Age: {data.traveler_info.age} years
Current Date: {current_date_str}
Days Until Departure: {days_until_departure} days
Travel Period Range: Day {travel_day_starts} to Day {travel_day_ends} from today
Destinations: {countries_list}
Trip Duration: {total_trip_days} days
Departure: {departure_date_str}
Return: {last_return_date_str}

{countries_details}

{aggregate_risks}

MEDICAL RESEARCH INSTRUCTIONS:
1. Search CDC and SSI websites for current vaccine and malaria recommendations for each destination
2. Consider traveler's age, risk factors, and time available before departure ({days_until_departure} days)
3. Consider the AGGREGATE risk factors - if ANY country has a particular risk, include recommendations
4. Provide specific, actionable recommendations based on authoritative sources
5. DO NOT use example vaccine names - only recommend vaccines actually needed for these specific destinations and risk factors

CRITICAL OUTPUT REQUIREMENTS:

You MUST return ONLY a valid JSON object (no markdown, no code blocks, no extra text). The JSON must match this exact structure:

{{
  "recommended_vaccines": [
    "Vaccine Name 1",
    "Vaccine Name 2"
  ],
  "malaria_prevention_required": true or false,
  "travel_summary": {{
    "destinations": "{countries_list}",
    "total_trip_duration_days": {total_trip_days},
    "days_until_departure": {days_until_departure},
    "rural_or_forest_areas": {str(any_rural_stay).lower()},
    "contact_with_locals": {str(any_close_contact_local_pop).lower()},
    "staying_with_locals": {str(any_staying_with_family).lower()},
    "animal_contact": {str(any_close_contact_animals).lower()},
    "risky_activities": {str(any_risky_activities).lower()},
    "departure_date": "{departure_date_str}",
    "final_return_date": "{last_return_date_str}",
    "traveler_age": {data.traveler_info.age}
  }},
  "vaccination_schedules": [
  {{
    "visit_number": 1,
    "timing_description": "Today or as soon as possible",
    "days_from_today": 0,
    "vaccines_to_administer": ["Hepatitis A (Dose 1)", "Hepatitis B (Dose 1)", "DiTe Booster"],
    "administration_notes": "Initial consultation - multiple vaccines can be given together",
    "overlap_warning": null
  }},
  {{
    "visit_number": 2,
    "timing_description": "28 days after first consultation",
    "days_from_today": 28,
    "vaccines_to_administer": ["Hepatitis A (Dose 2)", "Hepatitis B (Dose 2)", "Japanese Encephalitis (Dose 1)"],
    "administration_notes": null,
    "overlap_warning": "Contact clinic if this falls during travel dates"
  }}
 ],
  "vaccine_protections": [
    {{
      "vaccine_name": "Actual Vaccine Name",
      "protection_onset": "When protection begins (e.g., '2-4 weeks after first dose')",
      "full_protection": "When full protection achieved (e.g., '2 weeks after second dose')",
      "immunity_duration": "How long immunity lasts (e.g., 'At least 20 years with booster')",
      "booster_info": "Booster requirements if any"
    }}
  ],
  "malaria_protocol": {{
    "is_required": true or false,
    "medication_name": "Name and dosage if required",
    "start_timing": "When to start (e.g., '1-2 days before travel')",
    "stop_timing": "When to stop (e.g., '7 days after return')",
    "administration_instructions": "How to take it",
    "side_effects": "Common side effects",
    "additional_protection": "Other protective measures"
  }}
}}

CRITICAL FORMATTING RULES:
- Return ONLY pure JSON - no markdown code blocks, no backticks, no extra text
- NEVER use negative day numbers in timing_description
- Use phrases like "X days before departure" or "Today" or "X days after Dose N"
- For timing_description, use clear plain language
- For days_from_today, you MUST calculate an exact integer day number from today (Day 0)
- NEVER use null for days_from_today
- For vague timings like "1-6 weeks before departure", calculate the midpoint: (departure_day - 3.5 weeks)
- For relative timings like "X months after Dose 2", add the days together: (Dose 2 day + X months)
- Round to nearest integer if needed
- If a vaccine has multiple administration options (injection vs oral), include multiple schedule_options
- If any vaccine dose would fall during travel dates ({departure_date_str} to {last_return_date_str}) or ({travel_day_starts} to {travel_day_ends}), add an overlap_warning
- Only recommend vaccines that are ACTUALLY needed based on CDC/SSI guidelines for these specific destinations and risk profile
- DO NOT include placeholder or example vaccines
- Ensure all schedules are realistic given {days_until_departure} days until departure


CONSULTATION VISIT RULES:
- Group vaccines by consultation visit, not by individual vaccine
- Multiple vaccines can be administered in the same visit
- Calculate optimal visit timing based on vaccine requirements
- Specify which dose of each vaccine (e.g., "Hepatitis A (Dose 1)")
- Flag overlap_warning if any visit falls during travel dates
- Consider accelerated schedules if time before departure is limited
- CRITICAL: Consultation visits MUST be sorted by days_from_today in ascending order
- If Visit 3 occurs before Visit 2 chronologically, renumber them so they are in chronological order
- Never assign visit numbers out of chronological sequence

DAY NUMBER CALCULATION RULES:
- Consultation 1: days_from_today = 0 (always today)
- Consultation N: days_from_today = cumulative days from Day 0
- Example: "28 days after first consultation" → days_from_today = 28
- Example: "6 months after Dose 1" → days_from_today = 180 (6 × 30 days)
- Example: "1-6 weeks before departure" where departure is Day {days_until_departure}:
  → Calculate midpoint: {days_until_departure} - 3.5 weeks = {days_until_departure} - 24.5 days
  → Round to integer
- Example: "1-6 months after Dose 2 (Day 28)":
  → Calculate midpoint: 28 + 3.5 months = 28 + 105 days = 133

OVERLAP WARNING CALCULATION (MANDATORY):
- Current date: {current_date_str}
- Travel departure date: {departure_date_str}
- Travel return date: {last_return_date_str}
- For EACH consultation visit, calculate: visit_date = add days_from_today to {current_date_str}
- If visit_date falls between {departure_date_str} and {last_return_date_str} (inclusive), add overlap_warning
- For EACH consultation visit, calculate: visit_date = current_date + days_from_today
- If days_from_today falls in range [{travel_day_starts} to {travel_day_ends}], you MUST set overlap_warning with:
  * Format: "⚠️ TRAVEL CONFLICT: This consultation would fall on [calculate exact date from days_from_today], which is during your travel ({departure_date_str} to {last_return_date_str}). Contact the clinic BEFORE departure to reschedule or arrange administration abroad."
  * Calculate the exact date by adding days_from_today to {current_date_str}
  * Example: If days_from_today=40 and travel is Day 38-69, include: "This consultation would fall on [DATE], which conflicts with travel"
- If days_from_today < {travel_day_starts} OR days_from_today > {travel_day_ends}, set overlap_warning to null
- Post-travel boosters (e.g., Hepatitis A at 6-12 months) should NOT be flagged
- DO NOT use vague language like "Contact clinic if this falls during travel"
- ONLY include overlap_warning if the mathematical calculation proves conflict
- Include the specific calculated date in the warning message


VACCINE-SPECIFIC RULES:
- Some vaccines like Hepatitis A allow dose 2 to be given 6-12 months later (after travel is fine)
- Some vaccines like Japanese Encephalitis have both standard (28 days) and accelerated (7 days) schedules
- Some vaccines like Typhoid have both injection and oral capsule options
- Only flag overlap_warning with proper explanation and mentioning contact clinic for doses that MUST be given before/during travel (not post-travel boosters)

Output Language Rules: 
- The output must be in Danish language
- ALL text including section headers, descriptions, and notes MUST be in Danish
- Vaccine names can remain in their international medical names
- Dates should use Danish format (e.g., "15. november 2025")

Return ONLY the JSON object. Do not wrap it in markdown code blocks. No additional text before or after the JSON."""

    return prompt


def create_grounding_config():
    """Create Google Search grounding configuration focused on CDC and SSI"""
    return types.GoogleSearch()


# ============================================================================
# STEP 4: Parse Structured Output and Convert to Human-Readable Text
# ============================================================================

# def structured_to_readable(structured: StructuredHealthPlan) -> str:
#     """Convert structured data to human-readable text format"""
    
#     output = []
    
#     # Section 1: Recommended Vaccines
#     output.append("1. Recommended Vaccines & Malaria Prevention\n")
#     for vaccine in structured.recommended_vaccines:
#         output.append(f"   • {vaccine}")
#     output.append("\n")
    
#     # Section 2: Travel Summary
#     output.append("2. Summary of Your Travel Info\n")
#     summary = structured.travel_summary
#     output.append(f"   • Destinations: {summary.destinations}")
#     output.append(f"   • Total Trip Duration: {summary.total_trip_duration_days} days")
#     output.append(f"   • Days Until Departure: {summary.days_until_departure} days")
#     output.append(f"   • Rural or Forest Areas: {'Yes' if summary.rural_or_forest_areas else 'No'}")
#     output.append(f"   • Contact with Locals: {'Yes' if summary.contact_with_locals else 'No'}")
#     output.append(f"   • Staying with Locals: {'Yes' if summary.staying_with_locals else 'No'}")
#     output.append(f"   • Animal Contact: {'Yes' if summary.animal_contact else 'No'}")
#     output.append(f"   • Risky Activities: {'Yes' if summary.risky_activities else 'No'}")
#     output.append(f"   • Departure Date: {summary.departure_date}")
#     output.append(f"   • Final Return Date: {summary.final_return_date}")
#     output.append(f"   • Traveler Age: {summary.traveler_age} years")
#     output.append("\n")
    
#     # Section 3: Vaccination Schedule
#     output.append("3. Vaccination Schedule Plan\n")
#     for schedule in structured.vaccination_schedules:
#         output.append(f"\n{schedule.vaccine_name}:")
        
#         for option in schedule.schedule_options:
#             if len(schedule.schedule_options) > 1:
#                 output.append(f"\n   {option.option_name}:")
            
#             for dose in option.doses:
#                 output.append(f"   • Dose {dose.dose_number}: {dose.timing_description}")
            
#             if option.administration_notes:
#                 output.append(f"     Note: {option.administration_notes}")
        
#         if schedule.overlap_warning:
#             output.append(f"\n   ⚠️  {schedule.overlap_warning}")
        
#         output.append("")
#     output.append("")
    
#     # Section 4: Protection Timeline
#     output.append("4. Vaccine Protection Timeline\n")
#     for protection in structured.vaccine_protections:
#         text = f"\n{protection.vaccine_name}:\n"
#         text += f"   • Protection starts: {protection.protection_onset}\n"
#         text += f"   • Full protection: {protection.full_protection}\n"
#         text += f"   • Immunity duration: {protection.immunity_duration}"
#         if protection.booster_info:
#             text += f"\n   • Booster info: {protection.booster_info}"
#         output.append(text)
#         output.append("")
#     output.append("")
    
#     # Section 5: Malaria Protocol
#     output.append("5. Malaria Prevention Protocol\n")
#     malaria = structured.malaria_protocol
    
#     if malaria.is_required:
#         output.append(f"   Medication: {malaria.medication_name}\n")
#         output.append(f"   • When to Start: {malaria.start_timing}")
#         output.append(f"   • When to Stop: {malaria.stop_timing}\n")
#         output.append(f"   • How to Take: {malaria.administration_instructions}\n")
#         output.append(f"   • Common Side Effects: {malaria.side_effects}\n")
#         if malaria.additional_protection:
#             output.append(f"   • Additional Protection: {malaria.additional_protection}")
#     else:
#         output.append("   ✓ Good news! Malaria prophylaxis is not required for your destinations\n     based on current CDC and SSI guidelines.")
    
#     # return "\n".join(output)
#     return "<br>".join(output)
def structured_to_readable(structured: StructuredHealthPlan) -> str:
    """Convert structured data to HTML-formatted text for website display in Danish"""
    
    html_parts = []
    
    # Section 1: Recommended Vaccines
    html_parts.append("<h2>1. Anbefalede Vacciner & Malaria Forebyggelse</h2>")
    html_parts.append("<ul>")
    for vaccine in structured.recommended_vaccines:
        html_parts.append(f"<li>{vaccine}</li>")
    html_parts.append("</ul>")
    
    # Section 2: Travel Summary
    html_parts.append("<h2>2. Oversigt over Din Rejse Information</h2>")
    summary = structured.travel_summary
    html_parts.append("<ul>")
    html_parts.append(f"<li><strong>Destinationer:</strong> {summary.destinations}</li>")
    html_parts.append(f"<li><strong>Samlet Rejsevarighed:</strong> {summary.total_trip_duration_days} dage</li>")
    html_parts.append(f"<li><strong>Dage til Afrejse:</strong> {summary.days_until_departure} dage</li>")
    html_parts.append(f"<li><strong>Landlige eller Skovområder:</strong> {'Ja' if summary.rural_or_forest_areas else 'Nej'}</li>")
    html_parts.append(f"<li><strong>Kontakt med Lokale:</strong> {'Ja' if summary.contact_with_locals else 'Nej'}</li>")
    html_parts.append(f"<li><strong>Opholder sig hos Lokale:</strong> {'Ja' if summary.staying_with_locals else 'Nej'}</li>")
    html_parts.append(f"<li><strong>Dyrekontakt:</strong> {'Ja' if summary.animal_contact else 'Nej'}</li>")
    html_parts.append(f"<li><strong>Risikable Aktiviteter:</strong> {'Ja' if summary.risky_activities else 'Nej'}</li>")
    html_parts.append(f"<li><strong>Afrejsedato:</strong> {summary.departure_date}</li>")
    html_parts.append(f"<li><strong>Endelig Returneringsdato:</strong> {summary.final_return_date}</li>")
    html_parts.append(f"<li><strong>Rejsendes Alder:</strong> {summary.traveler_age} år</li>")
    html_parts.append("</ul>")
    
    # Section 3: Vaccination Schedule (by Consultation Visit)
    html_parts.append("<h2>3. Vaccinationsplan</h2>")
    for visit in structured.vaccination_schedules:
        html_parts.append(f"<h3>Konsultation {visit.visit_number}</h3>")
        html_parts.append(f"<p><strong>Tidspunkt:</strong> {visit.timing_description}</p>")
        
        html_parts.append("<p><strong>Vacciner at modtage:</strong></p>")
        html_parts.append("<ul>")
        for vaccine in visit.vaccines_to_administer:
            html_parts.append(f"<li>{vaccine}</li>")
        html_parts.append("</ul>")
        
        if visit.administration_notes:
            html_parts.append(f"<p><em>Bemærk: {visit.administration_notes}</em></p>")
        
        if visit.overlap_warning:
            html_parts.append(f"<p><strong>⚠️ {visit.overlap_warning}</strong></p>")

    # Section 4: Protection Timeline
    html_parts.append("<h2>4. Vaccine Beskyttelsestidslinje</h2>")
    for protection in structured.vaccine_protections:
        html_parts.append(f"<h3>{protection.vaccine_name}</h3>")
        html_parts.append("<ul>")
        html_parts.append(f"<li><strong>Beskyttelse starter:</strong> {protection.protection_onset}</li>")
        html_parts.append(f"<li><strong>Fuld beskyttelse:</strong> {protection.full_protection}</li>")
        html_parts.append(f"<li><strong>Immunitetsvarighed:</strong> {protection.immunity_duration}</li>")
        if protection.booster_info:
            html_parts.append(f"<li><strong>Booster information:</strong> {protection.booster_info}</li>")
        html_parts.append("</ul>")
    
    # Section 5: Malaria Protocol
    html_parts.append("<h2>5. Malaria Forebyggelsesprotokol</h2>")
    malaria = structured.malaria_protocol
    
    if malaria.is_required:
        html_parts.append(f"<p><strong>Medicin:</strong> {malaria.medication_name}</p>")
        html_parts.append("<ul>")
        html_parts.append(f"<li><strong>Hvornår skal man starte:</strong> {malaria.start_timing}</li>")
        html_parts.append(f"<li><strong>Hvornår skal man stoppe:</strong> {malaria.stop_timing}</li>")
        html_parts.append(f"<li><strong>Hvordan tages det:</strong> {malaria.administration_instructions}</li>")
        html_parts.append(f"<li><strong>Almindelige Bivirkninger:</strong> {malaria.side_effects}</li>")
        if malaria.additional_protection:
            html_parts.append(f"<li><strong>Yderligere Beskyttelse:</strong> {malaria.additional_protection}</li>")
        html_parts.append("</ul>")
    else:
        html_parts.append("<p>✓ Gode nyheder! Malaria profylakse er ikke påkrævet for dine destinationer baseret på nuværende CDC og SSI retningslinjer.</p>")
    
    return "".join(html_parts)


# ============================================================================
# STEP 5: Create Journal File
# ============================================================================

def create_journal_file(data: TravelRequest, health_plan: str, structured: StructuredHealthPlan) -> tuple:
    """
    Create a readable journal file with health recommendations.
    Returns: (filename, file_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    countries_abbr = "_".join([c.country_name[:3].upper() for c in data.countries])
    filename = f"Health_Plan_{countries_abbr}_{timestamp}.txt"
    file_path = JOURNALS_DIR / filename

    current_date = date.today()
    days_until_departure = (data.countries[0].departure_date - current_date).days

    countries_text = ""
    total_trip_days = 0
    for idx, country in enumerate(data.countries, 1):
        trip_days = country.duration_of_stay
        total_trip_days += trip_days

        risk_factors = []
        if country.rural_stay:
            risk_factors.append("staying in rural/forest areas")
        if country.close_contact_local_pop:
            risk_factors.append("close contact with local population")
        if country.staying_with_family:
            risk_factors.append("staying with local families")
        if country.close_contact_animals:
            risk_factors.append("potential animal contact")
        if country.risky_activities:
            risk_factors.append("risky activities like tattoos or healthcare work")

        risk_text = ", ".join(risk_factors) if risk_factors else "standard tourism activities"

        countries_text += f"""
Destination {idx}: {country.country_name}
You will be visiting {country.country_name} for {trip_days} days from {country.departure_date.strftime("%d %B %Y")} to {country.return_date.strftime("%d %B %Y")}. Your trip involves {risk_text}.

"""

    destinations_list = " and ".join([c.country_name for c in data.countries])

    journal_content = f"""
{'='*80}
                         TRAVEL HEALTH PLANNER JOURNAL
                           Your Personal Health Guide
{'='*80}

Report Generated: {datetime.now().strftime("%d %B %Y at %H:%M:%S")}


{'='*80}


{health_plan}


{'='*80}
FINAL WORDS
{'='*80}

This health plan is designed to help you travel safely and confidently. By following these recommendations and staying proactive about your health, you can focus on enjoying your journey to {destinations_list}.

Remember: Prevention is always better than treatment. Taking the time now to get properly vaccinated and prepared will give you peace of mind throughout your travels.

Safe travels and enjoy your adventure!

{'='*80}
"""

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(journal_content)
        logger.info(f"Journal file created: {file_path}")
        return filename, str(file_path)
    except Exception as e:
        logger.error(f"Error creating journal file: {e}")
        raise


# ============================================================================
# STEP 6: Create FastAPI App
# ============================================================================

app = FastAPI(
    title="Travel Health Planner API v5.0 - Structured Output",
    description="CDC & SSI based travel health recommendations with structured AI output (no bias from examples)",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],   # Allow all HTTP methods
    allow_headers=["*"],   # Allow all headers
)


# ============================================================================
# STEP 7: API Endpoints
# ============================================================================

@app.get("/")
def home():
    """Home page"""
    return {
        "message": "Welcome to Travel Health Planner API v5.0 - Structured Output",
        "version": "5.0.0",
        "based_on": ["CDC Yellow Book (Latest via Web)", "SSI Travel Vaccines (Latest via Web)"],
        "docs": "Visit http://localhost:8000/docs",
        "features": [
            "Structured JSON output (no bias from examples)",
            "Schema-enforced responses",
            "Single AI API call (50% faster & cheaper)",
            "Real-time web grounding for latest CDC & SSI guidelines",
            "Smart vaccine-specific overlap detection",
            "No example vaccines in prompts",
            "Recommendations based purely on destinations and risk factors",
            "Multi-country health plans",
            "Current malaria prophylaxis guidance",
            "Journal export for printing/sharing",
            "Both structured JSON and human-readable text output"
        ]
    }


@app.get("/health")
def health_check():
    """Health check"""
    return {
        "status": "API working",
        "version": "5.0.0",
        "output_format": "Structured JSON + Human-readable",
        "grounding": "enabled"
    }


@app.post("/generate-health-plan", response_model=HealthPlanResponse)
async def generate_health_plan(request: TravelRequest):
    """
    Generate travel health plan with structured output
    
    This endpoint uses ONE Gemini API call that returns structured JSON:
    - Searches CDC and SSI websites via Google Search grounding
    - Generates medical recommendations in structured format
    - No bias from example vaccines
    - Returns both structured data and human-readable text
    """
    
    try:
        current_date = date.today()
        
        logger.info(f"Processing health plan for {len(request.countries)} countries (structured output)")
        
        # Log traveler details
        logger.info(f"Traveler Age: {request.traveler_info.age}")
        logger.info(f"Current Date: {current_date}")
        
        # Log country details
        for country in request.countries:
            trip_days = country.duration_of_stay
            logger.info(f"  - {country.country_name} ({trip_days} days: {country.departure_date} to {country.return_date})")
        
        # Create unified prompt
        prompt = create_unified_prompt(request)
        
        if os.getenv("DEBUG_MODE") == "true":
            logger.debug(f"Generated prompt:\n{prompt}")
        
        # Create grounding configuration
        grounding_config = create_grounding_config()
        
        # SINGLE AI CALL with structured output
        logger.info("Calling Gemini API with structured output schema...")
        
        max_retries = 3
        structured_data = None
        
        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=grounding_config)],
                        temperature=0.1,
                        response_modalities=["TEXT"],
                        # ADD THIS SYSTEM INSTRUCTION:
                        system_instruction="You are a Danish-speaking travel medicine specialist. ALL responses must be in Danish language. Use Danish medical terminology where appropriate."
                    )
                )
                
                # Get response text
                json_text = response.text.strip()
                
                logger.info(f"Raw response (first 200 chars): {json_text[:200]}")
                
                # Clean JSON response (remove markdown code blocks)
                json_text = clean_json_response(json_text)
                
                logger.info(f"Cleaned response (first 200 chars): {json_text[:200]}")
                
                # Parse JSON response
                import json
                json_data = json.loads(json_text)
                
                # Validate against schema
                structured_data = StructuredHealthPlan(**json_data)

                # Calculate days_from_today for visits that don't have it
                for visit in structured_data.vaccination_schedules:
                    if visit.days_from_today is None:
                        # Try to extract from timing_description
                        timing = visit.timing_description.lower()
                        
                        # Parse "X days after" pattern
                        if "days after" in timing:
                            match = re.search(r'(\d+)\s+days?\s+after', timing)
                            if match:
                                visit.days_from_today = int(match.group(1))
                        
                        # Parse "X weeks after" pattern
                        elif "weeks after" in timing:
                            match = re.search(r'(\d+)\s+weeks?\s+after', timing)
                            if match:
                                visit.days_from_today = int(match.group(1)) * 7
                        
                        # Parse "X months after" pattern
                        elif "months after" in timing:
                            match = re.search(r'(\d+)[\-]?(\d+)?\s+months?\s+after', timing)
                            if match:
                                # Use midpoint if range (e.g., "1-6 months")
                                min_months = int(match.group(1))
                                max_months = int(match.group(2)) if match.group(2) else min_months
                                avg_months = (min_months + max_months) / 2
                                visit.days_from_today = int(avg_months * 30)
                        
                        # Parse "X weeks before departure" pattern
                        elif "before departure" in timing:
                            match = re.search(r'(\d+)[\-]?(\d+)?\s+weeks?\s+before', timing)
                            if match:
                                min_weeks = int(match.group(1))
                                max_weeks = int(match.group(2)) if match.group(2) else min_weeks
                                avg_weeks = (min_weeks + max_weeks) / 2
                                visit.days_from_today = days_until_departure - int(avg_weeks * 7)
                        
                        # If still null, put at end
                        if visit.days_from_today is None:
                            visit.days_from_today = 99999

                # Sort consultation visits by days_from_today
                structured_data.vaccination_schedules.sort(key=lambda visit: visit.days_from_today)

                # Renumber visits to be in order (1, 2, 3, etc.)
                for idx, visit in enumerate(structured_data.vaccination_schedules, start=1):
                    visit.visit_number = idx
                
                logger.info(f"✓ Successfully generated structured health plan on attempt {attempt + 1}")
                logger.info(f"  - Vaccines recommended: {len(structured_data.recommended_vaccines)}")
                logger.info(f"  - Malaria prevention: {structured_data.malaria_prevention_required}")
                break
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {e}")
                logger.error(f"Response text (first 500 chars): {response.text[:500]}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Failed to parse AI response as JSON after {max_retries} attempts. Error: {str(e)}"
                    )
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
        
        if not structured_data:
            raise HTTPException(status_code=500, detail="Failed to generate health plan after all retries")
        
        # Convert structured data to human-readable format
        human_readable = structured_to_readable(structured_data)
        
        # # Extract grounding metadata if available
        # sources_used = []
        # if hasattr(response, 'candidates') and response.candidates:
        #     candidate = response.candidates[0]
        #     if hasattr(candidate, 'grounding_metadata'):
        #         metadata = candidate.grounding_metadata
        #         if hasattr(metadata, 'grounding_chunks'):
        #             for chunk in metadata.grounding_chunks:
        #                 if hasattr(chunk, 'web'):
        #                     sources_used.append(chunk.web.uri)
        #         elif hasattr(metadata, 'web_search_queries'):
        #             # Log search queries used
        #             logger.info(f"Web searches performed: {metadata.web_search_queries}")

        # Extract grounding metadata if available

        # sources_used = []
        # if hasattr(response, 'candidates') and response.candidates:
        #     candidate = response.candidates[0]
        #     if hasattr(candidate, 'grounding_metadata'):
        #         metadata = candidate.grounding_metadata
        #         # Check if grounding_chunks exists AND is not None
        #         if hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks is not None:
        #             for chunk in metadata.grounding_chunks:
        #                 if hasattr(chunk, 'web'):
        #                     sources_used.append(chunk.web.uri)
        #         elif hasattr(metadata, 'web_search_queries'):
        #             # Log search queries used
        #             logger.info(f"Web searches performed: {metadata.web_search_queries}")
        
        # # Create journal file
        # filename, file_path = create_journal_file(request, human_readable, structured_data)
        
        # # Prepare response
        # return HealthPlanResponse(
        #     status="success",
        #     health_plan=human_readable,
        #     structured_data=structured_data,
        #     journal_file=filename,
        #     journal_download_path=f"/download-journal/{filename}",
        #     countries_analyzed=len(request.countries),
        #     traveler_age=request.traveler_info.age,
        #     sources_used=sources_used if sources_used else None
        # )
        # Create journal file
        # filename, file_path = create_journal_file(request, human_readable, structured_data)
        
        # Prepare response
        return HealthPlanResponse(
            health_plan=human_readable
        )

    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating health plan: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate health plan: {str(e)}"
        )


@app.get("/download-journal/{filename}")
async def download_journal(filename: str):
    """
    Download journal file
    
    Args:
        filename: Name of the journal file to download
    
    Returns:
        FileResponse with the journal file
    """
    file_path = JOURNALS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Journal file not found: {filename}"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/plain"
    )


@app.get("/journals")
async def list_journals():
    """
    List all available journal files
    
    Returns:
        List of journal filenames with metadata
    """
    try:
        journals = []
        for file_path in JOURNALS_DIR.glob("*.txt"):
            stat = file_path.stat()
            journals.append({
                "filename": file_path.name,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "size_bytes": stat.st_size,
                "download_path": f"/download-journal/{file_path.name}"
            })
        
        # Sort by creation time (newest first)
        journals.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "status": "success",
            "total_journals": len(journals),
            "journals": journals
        }
    
    except Exception as e:
        logger.error(f"Error listing journals: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list journals: {str(e)}"
        )


# ============================================================================
# STEP 8: Run the Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Travel Health Planner API v5.0...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )