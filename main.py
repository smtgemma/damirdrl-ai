import os
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from google import genai
from google.genai import types

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
    traveler_info: TravelerInfo
    countries: List[CountryData] = Field(..., min_items=1, max_items=10)


class HealthPlanResponse(BaseModel):
    """Response with health plan"""
    status: str
    health_plan: str
    journal_file: str
    journal_download_path: str
    countries_analyzed: int
    traveler_age: int
    sources_used: Optional[List[str]] = None ###############################################################################


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
# STEP 3: Create the Prompt with Web Grounding
# ============================================================================

def create_prompt(data: TravelRequest) -> str:
    """Create a prompt for Gemini with specific website grounding"""
    
    # Get current date
    current_date = date.today()
    
    # Build detailed country information
    countries_details = ""
    total_trip_days = 0
    
    for idx, country in enumerate(data.countries, 1):
        trip_duration = country.duration_of_stay
        total_trip_days += trip_duration
        countries_details += f"""
COUNTRY {idx}: {country.country_name}
---
Trip Duration: {trip_duration} days (from {country.departure_date} to {country.return_date})
Risk Factors:
- Rural/forest areas: {'Yes' if country.rural_stay else 'No'}
- Close contact with locals: {'Yes' if country.close_contact_local_pop else 'No'}
- Staying with local family: {'Yes' if country.staying_with_family else 'No'}
- Animal contact: {'Yes' if country.close_contact_animals else 'No'}
- Risky activities (tattoos/surgery/healthcare work): {'Yes' if country.risky_activities else 'No'}
"""
    
    countries_list = ", ".join([c.country_name for c in data.countries])
    current_date_str = current_date.strftime("%d %b %Y")
    departure_date_str = data.countries[0].departure_date.strftime("%d %b %Y")
    last_return_date_str = data.countries[-1].return_date.strftime("%d %b %Y")
    
    # Calculate days until departure
    days_until_departure = (data.countries[0].departure_date - current_date).days
    
    prompt = f"""
You are a travel health expert with access to the latest CDC and SSI guidelines through web search.

CRITICAL SEARCH INSTRUCTIONS:
You MUST search and consult BOTH of these authoritative sources:

1. CDC (US Centers for Disease Control):
   - Search: "CDC travel health {countries_list} vaccines"
   - Website: wwwnc.cdc.gov/travel/destinations
   - Look for country-specific vaccine recommendations

2. SSI (Statens Serum Institut - Denmark):
   - Search: "SSI Denmark rejse.ssi.dk {countries_list} vaccination"
   - Website: rejse.ssi.dk
   - This is the official Danish health authority for travel medicine


TRAVELER PROFILE:
- Age: {data.traveler_info.age} years
- Today: {current_date_str}
- Destinations: {countries_list}
- Number of Countries: {len(data.countries)}
- First Departure: {departure_date_str} (in {days_until_departure} days)
- Final Return: {last_return_date_str}
- Total Trip Duration: {total_trip_days} days

{countries_details}

RESEARCH INSTRUCTIONS:
For EACH destination, you must search:
1. "CDC travel {country.country_name} vaccines and malaria" 
2. "SSI rejse.ssi.dk {country.country_name} vaccination and malaria" 
3. Look for country-specific pages on both CDC and SSI websites
4. Check for disease outbreaks or health alerts
5. Verify age-specific recommendations from both authorities
6. Find current vaccination schedules and timing from both sources
7. malaria prophylaxis recommendations from CDC vs SSI



CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

1. Base ALL recommendations on current CDC AND SSI guidelines found through search
2. Marge the information from BOTH sources for accuracy
3. Consider trip duration, activities, and risk factors for EACH country
4. Provide combined recommendations considering all {len(data.countries)} destinations without losing detail
5. The vaccination schedule MUST be apointment-based using proper day gaps (Appoinment 1: Day 0, Appoinment 2: Day 7, etc.)
6. The vaccination schedule MUST use accelerated schedules if time until departure is limited
7. If a vaccine dose falls during travel dates, note: "Contact clinic - for doctor's advice"
8. Format response with EXACT structure shown below:

---RESPONSE FORMAT (Use this exact structure)---

1. Recommended Vaccines & Malaria Prevention

IMPORTANT SCHEDULING NOTES:
- Include only the names of recommended travel-specific vaccines and malaria medications.
- The Ai needs to research the latest recommendations from CDC and SSI for the given countries and given risks.
- Present them using alphabetical points (a, b, c, etc.).
- Do not add any extra details, descriptions, sources, or dosages.
- Do not use tables.
- Don't print instructions or extra text, just the content in given format.

**Vaccine List Example**
a. Hepatitis A
b. Typhoid
c. Yellow Fever
d. Japanese Encephalitis
e. Rabies
f. Meningococcal ACWY
g. Cholera
h. Malaria prophylaxis (e.g., Malarone, Doxycycline, Mefloquine)


2. Summary of Your Travel Info

IMPORTANT SCHEDULING NOTES:
- Generate the section titled "2. Summary of Your Travel Info" in a journal-style format.
- Present all information in a clear, point-to-point textual format.
- Do not use tables.
- Use complete sentences or short descriptive lines for readability.
- Don't print instructions or extra text, just the content in given format.

**Content**
- Destinations: {countries_list}
- Total Trip Duration: {total_trip_days} days
- Days Until Departure: {days_until_departure} days
- Rural or Forest Areas: {'Yes' if any(c.rural_stay for c in data.countries) else 'No'}
- Contact with Locals: {'Yes' if any(c.close_contact_local_pop for c in data.countries) else 'No'}
- Staying with Locals: {'Yes' if any(c.staying_with_family for c in data.countries) else 'No'}
- Animal Contact: {'Yes' if any(c.close_contact_animals for c in data.countries) else 'No'}
- Risky Activities: {'Yes' if any(c.risky_activities for c in data.countries) else 'No'}
- Departure Date: {departure_date_str}
- Final Return Date: {last_return_date_str}
- Traveler Age: {data.traveler_info.age}








3. Vaccination Schedule Plan

**Pre-Departure Vaccinations**
- List vaccines by Day numbers relative to today (Day 0 = today, Day 7 = 7 days from today, etc.)
- For accelerated schedules, always include in brackets: e.g., "Japanese Encephalitis (accelerated schedule dose 1)"
- Use standard intervals or accelerated schedules based on {days_until_departure}

**Dose Scheduling Logic (MUST FOLLOW):**

For each multi-dose vaccine:
1. Calculate each dose date: Dose date = Day 0 + interval for that dose
2. Apply these decision rules in order:

   **Rule A: All doses can be completed before departure**
   - IF all doses ≤ {days_until_departure}
   - THEN schedule normally in Pre-Departure with proper day numbers
   - Use accelerated schedule if time is limited but all doses still fit

   **Rule B: ANY dose falls during travel (CRITICAL)**
   - IF any dose > {days_until_departure} AND ≤ ({days_until_departure} + {total_trip_days})
   - THEN for that vaccine:
     * List ALL doses on their calculated days
     * Mark EACH dose that falls during travel with: "[Vaccine name] (dose X) - Contact clinic - dose needed during travel"
     * Also mark accelerated doses: "[Vaccine name] (accelerated schedule dose X) - Contact clinic - dose needed during travel"
     * This applies even if only 1 dose falls during travel - the entire vaccine series must be flagged

   **Rule C: All doses fall after return**
   - IF all doses > ({days_until_departure} + {total_trip_days})
   - THEN schedule entire series in Post-Return section

**Example Format:**

Pre-Departure:
- Day 0: Hepatitis A, Typhoid, Tdap, Hepatitis B (dose 1), Japanese Encephalitis (accelerated schedule dose 1), Rabies (dose 1), Malaria prophylaxis (start)
- Day 7: Hepatitis B (dose 2), Japanese Encephalitis (accelerated schedule dose 2), Rabies (dose 2)
- Day 14: Rabies (dose 3)
- Day 21: Hepatitis B (dose 3) - Contact clinic - dose needed during travel
- Day 28: Japanese Encephalitis (accelerated schedule dose 3) - Contact clinic - dose needed during travel

**Post-Return Vaccinations**
- Include ONLY doses scheduled after {last_return_date_str}
- Number sequentially as "Day X after return" (starting from Day 1)
- Do NOT include any doses that fall during travel
- Format:
  * Day 1 after return: [Vaccine name] - Complete series / Booster
  * Day 2 after return: [Vaccine name] - Complete series / Booster

**Critical Parameters:**
- {days_until_departure} = days from today until departure date
- {total_trip_days} = total duration of trip in days
- Last return date: {last_return_date_str}

**Mandatory Requirements:**
✓ ALL accelerated schedule doses MUST be shown in brackets
✓ ALL doses falling during travel MUST include "Contact clinic - dose needed during travel"
✓ If even 1 dose of a vaccine falls during travel, flag that vaccine's name for clinic contact
✓ Use both standard and accelerated schedules in calculations to maximize vaccination options
✓ Multi-dose vaccines must show all doses with proper day spacing
✓ Only list vaccines that can reasonably be started before departure
✓ Never schedule in-travel doses in Post-Return section

**Calculation Method:**
For each vaccine, calculate using BOTH:
1. Standard dosing intervals (e.g., 0, 28, 180 days)
2. Accelerated dosing intervals (e.g., 0, 7, 21 days)
Choose the schedule that best fits the available time while following Rules A, B, and C above.












4. Vaccine Protection Timeline

IMPORTANT VACCINE PROTECTION NOTES:

- Generate the section titled "4. Vaccine Protection Timeline".
- Present all vaccines in a clear, point-to-point textual format.
- Do not use tables.
- Include protection start time, full protection timing, and duration of immunity for each vaccine.

EXAMPLE:
- Hepatitis A: Protection begins 2-4 weeks after 1st dose; full protection after 2nd dose (for long-term); duration of immunity 20+ years with 2 doses
- Japanese Encephalitis: Protection begins 7-10 days after 2nd dose; full protection after 2nd dose (standard) or 3rd dose (extended); duration of immunity 1-2 years, booster needed
- Typhoid (injection): Protection begins 2 weeks after dose; full protection after single dose; duration of immunity 2-3 years
- Rabies: Protection begins after 3rd dose; full protection after 3-dose series complete; duration of immunity 2-3 years

5. Malaria Prevention Protocol

IF MALARIA RISK EXISTS:
- Generate the section titled "5. Malaria Prevention Protocol".
- Present all information in a clear, point-to-point textual format.
- Do not use tables.
- Include medication name, dosage, start and stop timing, administration instructions, side effects, contraindications, and schedule.
- If no malaria risk exists, explicitly state that prophylaxis is not required.


**Medication Details**

IF MALARIA RISK EXISTS:
- Medication Name: [Specific drug - Malarone/Doxycycline/Mefloquine]
- Dosage: [Exact dose, e.g., "1 tablet daily"]
- When to Start: [X days before entering malaria zone]
- When to Stop: [X days/weeks after leaving malaria zone]
- How to Take: [With food/water, time of day]
- Side Effects: [Common side effects]
- Contraindications: [Who should avoid]
- Schedule:
  - Start: {days_until_departure} day X (based on "start before" guidance)
  - Continue throughout trip: {total_trip_days} days
  - Stop: Day X after return (based on "continue after" guidance)
  - Total duration: [Calculate total days]


IF NO MALARIA RISK:
- Malaria prophylaxis is not required for {countries_list} based on current CDC and SSI guidelines.


6. Essential Health Precautions

**Food & Water Safety**
[Specific guidance]

**Insect Bite Prevention**
[Specific guidance for vector-borne diseases]

**Travel Insurance**
[Recommendation with specifics]

**When to Seek Medical Help**
[Warning signs and symptoms]

**Current Health Alerts**
[Any active outbreaks or concerns for destinations]

**Additional Precautions**
[Based on risk factors identified]


---

Now generate the response in the exact format above using current information from web search. Remember:
- Use only vaccine NAMES in section 1 
- Use appointment DAY numbers (Day 0, Day 7, etc.) in section 3, NOT specific dates
- If vaccine timing conflicts with travel, note "Contact clinic" for that appointment
- Use your knowledge of standard vaccine intervals and accelerated schedules
"""
    return prompt


def create_grounding_config():
    """Create Google Search grounding configuration focused on CDC and SSI"""
    return types.GoogleSearch()


# ============================================================================
# STEP 4: Create Journal File
# ============================================================================

def create_journal_file(data: TravelRequest, health_plan: str) -> tuple:
    """
    Create a readable journal file with health recommendations in simple prose format
    
    Returns:
        tuple: (filename, file_path)
    """
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    countries_abbr = "_".join([c.country_name[:3].upper() for c in data.countries])
    filename = f"Health_Plan_{countries_abbr}_{timestamp}.txt"
    file_path = JOURNALS_DIR / filename
    
    # Get current date
    current_date = date.today()
    days_until_departure = (data.countries[0].departure_date - current_date).days
    
    # Build country descriptions
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
You'll be visiting {country.country_name} for {trip_days} days from {country.departure_date.strftime("%d %B %Y")} to {country.return_date.strftime("%d %B %Y")}. Your trip involves {risk_text}.

"""
    
    destinations_list = " and ".join([c.country_name for c in data.countries])
    
    # Create journal content in prose format
    journal_content = f"""
{'='*80}
                         TRAVEL HEALTH PLANNER JOURNAL
                           Your Personal Health Guide
{'='*80}

Report Generated: {datetime.now().strftime("%d %B %Y at %H:%M:%S")}

{'='*80}
ABOUT YOU AND YOUR TRIP
{'='*80}

Hello! This personalized health journal has been created specifically for your upcoming travel to {destinations_list}. 

TRAVELER PROFILE
You are {data.traveler_info.age} years old, and this report was prepared on {current_date.strftime("%d %B %Y")}. You have {days_until_departure} days until your departure, so it's important to start your vaccination schedule as soon as possible.

YOUR JOURNEY
You'll be traveling to {len(data.countries)} {"destination" if len(data.countries) == 1 else "destinations"} over a total period of {total_trip_days} days.

{countries_text}

{'='*80}
YOUR PERSONALIZED HEALTH RECOMMENDATIONS
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
    
    # Write to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(journal_content)
        logger.info(f"Journal file created: {file_path}")
        return filename, str(file_path)
    except Exception as e:
        logger.error(f"Error creating journal file: {e}")
        raise


# ============================================================================
# STEP 5: Create FastAPI App
# ============================================================================

app = FastAPI(
    title="Travel Health Planner API with Web Grounding",
    description="CDC & SSI based personalized travel health recommendations with real-time web research",
    version="3.1.0"
)


# ============================================================================
# STEP 6: API Endpoints
# ============================================================================

@app.get("/")
def home():
    """Home page"""
    return {
        "message": "Welcome to Travel Health Planner API v3.1 with Web Grounding",
        "version": "3.1.0",
        "based_on": ["CDC Yellow Book (Latest via Web)", "SSI Travel Vaccines (Latest via Web)"],
        "docs": "Visit http://localhost:8000/docs",
        "features": [
            "Real-time web grounding for latest CDC & SSI guidelines",
            "Google Search integration for current health alerts",
            "Multi-country health plans",
            "Day-based vaccination schedules (Day 0, Day 7, etc.)",
            "Accelerated vaccine schedules when needed",
            "Current malaria prophylaxis guidance",
            "Risk assessment analysis",
            "Journal export for printing/sharing",
            "Source citation with URLs",
            "Up-to-date disease outbreak information",
            "Improved table formatting"
        ]
    }


@app.get("/health")
def health_check():
    """Health check"""
    return {"status": "API working", "version": "3.1.0", "grounding": "enabled"}


@app.post("/generate-health-plan", response_model=HealthPlanResponse)
async def generate_health_plan(request: TravelRequest):
    """
    Generate travel health plan with real-time web grounding
    
    This endpoint uses Google Search to access the latest information from:
    - CDC Travel Health Notices (wwwnc.cdc.gov/travel)
    - SSI Denmark Travel Vaccines (rejse.ssi.dk)
    
    ALL FIELDS ARE REQUIRED for accurate recommendations.
    
    Response includes:
    - health_plan: AI-generated recommendations with current data
    - journal_file: Filename of the generated journal
    - journal_download_path: URL to download the journal
    - sources_used: List of URLs consulted (when available)
    """
    
    try:
        # Get current date
        current_date = date.today()
        
        logger.info(f"Processing health plan for {len(request.countries)} countries with web grounding")
        
        # Log traveler details
        logger.info(f"Traveler Age: {request.traveler_info.age}")
        logger.info(f"Current Date: {current_date}")
        
        # Log country details
        for country in request.countries:
            trip_days = country.duration_of_stay
            logger.info(f"  - {country.country_name} ({trip_days} days: {country.departure_date} to {country.return_date})")
        
        # Create the prompt
        prompt = create_prompt(request)
        
        if os.getenv("DEBUG_MODE") == "true":
            logger.debug(f"Generated prompt:\n{prompt}")
        
        # Create grounding configuration
        grounding_config = create_grounding_config()
        
        # Send to Gemini with grounding enabled
        logger.info("Calling Gemini API with Google Search grounding...")
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=grounding_config)],
                temperature=0.2,
            )
        )
        
        logger.info("✓ Successfully generated health plan with web grounding")
        
        # Extract grounding metadata if available
        sources_used = []
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding_meta = candidate.grounding_metadata
                if hasattr(grounding_meta, 'search_entry_point'):
                    logger.info("✓ Grounding metadata found - sources were consulted")
        
        # Create journal file
        filename, file_path = create_journal_file(request, response.text)
        
        logger.info(f"✓ Journal file created: {filename}")
        
        # Return the result
        return HealthPlanResponse(
            status="success",
            health_plan=response.text,
            journal_file=filename,
            journal_download_path=f"/download-journal/{filename}",
            countries_analyzed=len(request.countries),
            traveler_age=request.traveler_info.age,
            sources_used=sources_used if sources_used else None
        )
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/download-journal/{filename}")
def download_journal(filename: str):
    """Download journal file"""
    try:
        file_path = JOURNALS_DIR / filename
        
        # Security check - ensure file is in journals directory
        if not file_path.resolve().is_relative_to(JOURNALS_DIR.resolve()):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Journal file not found")
        
        logger.info(f"Downloading journal: {filename}")
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='text/plain'
        )
    except Exception as e:
        logger.error(f"Error downloading journal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-journals")
def list_journals():
    """List all generated journals"""
    try:
        journals = sorted([f.name for f in JOURNALS_DIR.glob("*.txt")], reverse=True)
        return {
            "total_journals": len(journals),
            "journals": journals,
            "directory": str(JOURNALS_DIR)
        }
    except Exception as e:
        logger.error(f"Error listing journals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# STEP 7: Example Data for Testing
# ============================================================================

@app.get("/example-thailand-vietnam")
def example_thailand_vietnam():
    """Multi-country example: Thailand + Vietnam (high risk)"""
    return {
        "traveler_info": {
            "age": 29
        },
        "countries": [
            {
                "country_name": "Thailand",
                "departure_date": "2025-11-11",
                "return_date": "2025-11-25",
                "rural_stay": True,
                "close_contact_local_pop": True,
                "staying_with_family": True,
                "close_contact_animals": True,
                "risky_activities": False
            },
            {
                "country_name": "Vietnam",
                "departure_date": "2025-11-26",
                "return_date": "2025-12-17",
                "rural_stay": True,
                "close_contact_local_pop": True,
                "staying_with_family": False,
                "close_contact_animals": False,
                "risky_activities": False
            }
        ]
    }


@app.get("/example-bangladesh")
def example_bangladesh():
    """Single country example: Bangladesh"""
    return {
        "traveler_info": {
            "age": 35
        },
        "countries": [
            {
                "country_name": "Bangladesh",
                "departure_date": "2025-11-15",
                "return_date": "2025-11-25",
                "rural_stay": True,
                "close_contact_local_pop": True,
                "staying_with_family": False,
                "close_contact_animals": False,
                "risky_activities": False
            }
        ]
    }


# ============================================================================
# STEP 8: Startup Message
# ============================================================================

@app.on_event("startup")
def startup():
    logger.info("=" * 80)
    logger.info("Travel Health Planner API v3.1 Starting...")
    logger.info("=" * 80)
    logger.info("🔬 Based on: CDC & SSI (via Real-Time Web Grounding)")
    logger.info("🌐 Google Search: Enabled")
    logger.info("📍 API Documentation: http://localhost:8000/docs")
    logger.info("📝 Journal Export: Enabled (saved to ./journals/)")
    logger.info("🔍 Dynamic Retrieval: Active for latest health data")
    logger.info("📅 Schedule Format: Day-based (Day 0, Day 7, etc.)")
    logger.info("=" * 80)


# Run this file with: uvicorn main:app --reload