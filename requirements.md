# Suraksha-Path - Requirements Document

## Project Overview

### Mission
Suraksha-Path is an AI-powered navigation system that prioritizes personal safety over shortest distance for night-time travel. The platform analyzes multiple safety parameters to recommend the safest routes, empowering users to make informed decisions about their travel paths.

### Target Users
- Women traveling alone during evening/night hours
- Elderly individuals requiring safer navigation options
- Solo travelers in unfamiliar areas
- Anyone prioritizing safety over speed in urban navigation

### Core Value Proposition
Unlike traditional navigation apps that optimize for time and distance, Suraksha-Path optimizes for safety by analyzing street lighting, crowd density, historical incident data, and real-time factors to suggest the safest possible routes.

---

## Functional Requirements

### 1. Safety Score Calculation
**Priority: High**

- Calculate a safety score (0-100) for each route segment based on multiple parameters
- Display aggregate safety score for complete routes
- Provide breakdown of safety factors contributing to the score
- Update scores dynamically based on time of day
- Color-code route segments (Green: Safe, Yellow: Moderate, Red: Unsafe)

**Acceptance Criteria:**
- Safety score must be calculated within 2 seconds for routes up to 10km
- Score must consider minimum 4 safety parameters
- Visual representation must be intuitive and accessible

### 2. Route Comparison (Fastest vs. Safest)
**Priority: High**

- Display minimum 2 route options: Fastest and Safest
- Show comparative metrics for each route:
  - Total distance
  - Estimated time
  - Safety score
  - Key safety highlights/concerns
- Allow users to view intermediate route options
- Enable route customization by avoiding specific areas

**Acceptance Criteria:**
- Side-by-side comparison view
- Clear visual differentiation between route types
- Time difference and safety score difference prominently displayed

### 3. Real-time Location Sharing
**Priority: High**

- Enable users to share live location with trusted contacts
- Support sharing via SMS, WhatsApp, or in-app link
- Allow time-bound sharing (auto-expire after journey completion)
- Show ETA to trusted contacts
- Send notifications when user deviates from planned route
- Provide "I've reached safely" quick action button

**Acceptance Criteria:**
- Location updates every 30 seconds during active sharing
- Shareable link works without app installation
- Maximum 5-second delay in location updates

### 4. SOS Trigger
**Priority: Critical**

- One-tap SOS button accessible from all screens
- Trigger actions on SOS activation:
  - Send location to emergency contacts
  - Alert local authorities (integration with emergency services)
  - Start audio/video recording (with user consent)
  - Send pre-configured emergency message
- Shake-to-activate SOS as alternative trigger
- False alarm cancellation with 5-second window

**Acceptance Criteria:**
- SOS activation must complete within 1 second
- Minimum 3 emergency contacts supported
- Works in low/no network conditions (queue for later transmission)

### 5. Safety Alerts and Notifications
**Priority: Medium**

- Real-time alerts for entering low-safety zones
- Notifications about nearby safe spaces (police stations, hospitals, 24/7 establishments)
- Weather-based safety alerts
- Community-reported incident alerts

### 6. Offline Functionality
**Priority: Medium**

- Download maps for offline use
- Cache safety scores for frequently traveled routes
- Basic navigation without internet connectivity
- Sync data when connection restored

---

## Non-Functional Requirements

### 1. Performance
- Route calculation latency: < 3 seconds for routes up to 15km
- App launch time: < 2 seconds
- Safety score computation: < 1 second per route segment
- Map rendering: 60 FPS on mid-range devices
- Support for 10,000+ concurrent users

### 2. Data Privacy and Security
- End-to-end encryption for location sharing
- No storage of user location history without explicit consent
- GDPR and Indian data protection compliance
- Anonymous data collection for safety analytics
- User data deletion on request within 48 hours
- Secure storage of emergency contact information
- No third-party data sharing without consent

### 3. Reliability and Availability
- 99.5% uptime for core services
- Graceful degradation when external APIs fail
- Offline mode for critical features
- Automatic retry mechanism for failed requests

### 4. Scalability
- Horizontal scaling capability for increased user load
- Efficient caching strategy to reduce API calls
- Database optimization for quick safety score retrieval

### 5. Usability
- Intuitive UI requiring minimal learning curve
- Accessibility compliance (screen reader support, high contrast mode)
- Multi-language support (Hindi, English, regional languages)
- Maximum 3 taps to access any core feature

### 6. Compatibility
- Android 8.0+ and iOS 13+ support
- Responsive web application
- Support for devices with 2GB+ RAM
- Works on 3G/4G/5G networks

---

## Safety Score Parameters

### 1. Street Lighting Analysis
**Weight: 30%**

**Data Source:** Google Street View images, OpenStreetMap lighting tags

**Methodology:**
- Use Computer Vision (CV) model to analyze street view images
- Detect presence and intensity of street lights
- Classify lighting levels: Well-lit (score: 90-100), Moderately lit (60-89), Poorly lit (30-59), Dark (0-29)
- Consider time of day (lighting score increases during daylight)

**CV Model Approach:**
- Pre-trained CNN (ResNet50/EfficientNet) for feature extraction
- Fine-tune on street lighting dataset
- Binary classification: Adequate lighting vs. Inadequate lighting
- Brightness histogram analysis as supplementary metric

**Implementation:**
```
Lighting Score = (Detected_Lights_Count × Light_Intensity × Time_Factor) / Route_Segment_Length
```

### 2. Crowd Density Analysis
**Weight: 25%**

**Data Source:** Google Places API, OpenStreetMap POI data, Mobile network data (if available)

**Methodology:**
- Analyze density of Points of Interest (POIs):
  - High-value POIs: Police stations, hospitals, 24/7 shops, restaurants
  - Medium-value POIs: Residential areas, parks with activity
  - Low-value POIs: Industrial areas, isolated zones
- Consider time-based crowd patterns
- Higher crowd density in well-populated areas = higher safety score

**Scoring Logic:**
```
Crowd Score = (High_Value_POI_Count × 3 + Medium_Value_POI_Count × 2) × Time_Activity_Multiplier
```

**Time Activity Multiplier:**
- 6 AM - 10 PM: 1.0 (normal activity)
- 10 PM - 12 AM: 0.7 (reduced activity)
- 12 AM - 6 AM: 0.4 (minimal activity)

### 3. Historical Incident Mapping
**Weight: 25%**

**Data Source:** Public crime databases, police records, crowdsourced incident reports

**Methodology:**
- Map historical incidents (theft, assault, harassment) to geographic coordinates
- Apply time-decay factor (recent incidents weighted higher)
- Create heat map of incident density
- Higher incident density = lower safety score

**Scoring Logic:**
```
Incident Score = 100 - (Incident_Count_Last_6_Months × Severity_Weight × Proximity_Factor)
```

**Severity Weights:**
- Critical incidents (assault, robbery): 10
- Moderate incidents (harassment, theft): 5
- Minor incidents (suspicious activity): 2

### 4. Road Type and Infrastructure
**Weight: 10%**

**Data Source:** OpenStreetMap road classifications

**Methodology:**
- Main roads and highways: Higher safety score
- Narrow lanes and alleys: Lower safety score
- Pedestrian-friendly infrastructure: Bonus points
- CCTV coverage: Bonus points (if data available)

### 5. Proximity to Safe Spaces
**Weight: 10%**

**Data Source:** Google Places API, Government databases

**Methodology:**
- Calculate distance to nearest:
  - Police station
  - Hospital/Medical facility
  - Fire station
  - 24/7 commercial establishment
- Closer proximity = higher safety score

**Scoring Logic:**
```
Safe_Space_Score = 100 - (Distance_to_Nearest_Safe_Space_in_meters / 50)
```

### Aggregate Safety Score Calculation
```
Final_Safety_Score = (Lighting_Score × 0.30) + 
                     (Crowd_Score × 0.25) + 
                     (Incident_Score × 0.25) + 
                     (Road_Infrastructure_Score × 0.10) + 
                     (Safe_Space_Proximity_Score × 0.10)
```

---

## Tech Stack Recommendations

### Backend
- **Language:** Python 3.9+
- **Framework:** FastAPI (high performance, async support)
- **Database:** 
  - PostgreSQL with PostGIS extension (geospatial queries)
  - Redis (caching, session management)
- **Task Queue:** Celery with Redis broker (async processing)
- **API Gateway:** Kong or AWS API Gateway

### Frontend
- **Web:** React.js with TypeScript
- **Mobile:** React Native or Flutter (cross-platform)
- **State Management:** Redux Toolkit or Zustand
- **UI Components:** Material-UI or Tailwind CSS

### Mapping and Geospatial
- **Base Maps:** 
  - OpenStreetMap (free, open-source)
  - Google Maps API (fallback, better coverage)
- **Map Rendering:** 
  - Leaflet.js (lightweight, open-source)
  - Mapbox GL JS (advanced features, better performance)
- **Routing Engine:** 
  - OSRM (Open Source Routing Machine)
  - GraphHopper (flexible, supports custom routing)
- **Geocoding:** Nominatim (OSM) or Google Geocoding API

### AI/ML Components
- **Computer Vision:** 
  - TensorFlow 2.x or PyTorch 1.13+
  - Pre-trained models: EfficientNet, ResNet50, MobileNetV3
  - OpenCV for image preprocessing
- **Model Serving:** TensorFlow Serving or TorchServe
- **ML Pipeline:** MLflow for experiment tracking

### Data Processing
- **ETL:** Apache Airflow
- **Data Analysis:** Pandas, NumPy, GeoPandas
- **Visualization:** Matplotlib, Plotly

### Infrastructure
- **Cloud Platform:** AWS, Google Cloud, or Azure
- **Containerization:** Docker
- **Orchestration:** Kubernetes (for production scale)
- **CI/CD:** GitHub Actions or GitLab CI
- **Monitoring:** Prometheus + Grafana, Sentry for error tracking

### APIs and Services
- **Google Maps Platform:**
  - Maps JavaScript API
  - Directions API
  - Places API
  - Street View Static API
- **OpenStreetMap:**
  - Overpass API (POI data)
  - Nominatim (geocoding)
- **Communication:**
  - Twilio (SMS for SOS)
  - Firebase Cloud Messaging (push notifications)

---

## Data Sources

### 1. Crime and Incident Data

**Public Datasets:**
- **National Crime Records Bureau (NCRB):** Annual crime statistics by state/city
  - URL: https://ncrb.gov.in/
  - Format: PDF reports (requires extraction)
  
- **Open Government Data (OGD) Platform India:** Various datasets including crime data
  - URL: https://data.gov.in/
  - Search for: "crime statistics", "police records"

- **City-specific Police Portals:** Some cities publish crime data
  - Delhi Police: Crime mapping initiatives
  - Mumbai Police: Public safety data

**Synthetic/Alternative Data:**
- **Crowdsourced Reports:** Build community reporting feature
- **News Scraping:** Extract incident data from local news (with proper attribution)
- **Social Media Analysis:** Twitter/X mentions of incidents (anonymized)
- **Kaggle Datasets:** 
  - "Crime Data from 2020 to Present" (Los Angeles)
  - "India Crime Data" (various sources)
  - Adapt and augment for Indian context

**Data Collection Strategy:**
- Start with synthetic data for MVP
- Partner with local police departments for pilot cities
- Implement crowdsourcing with verification mechanism
- Use historical news archives for incident mapping

### 2. Street View Images

**Primary Sources:**
- **Google Street View Static API:** 
  - Coverage: Major Indian cities
  - Cost: $7 per 1000 requests (consider budget)
  - Rate limits: Apply caching strategy

- **Mapillary:** Open street-level imagery
  - URL: https://www.mapillary.com/
  - API: Free for non-commercial use
  - Coverage: Growing in India

**Alternative Approaches:**
- **OpenStreetCam:** Community-contributed street images
- **Custom Data Collection:** Partner with delivery services or conduct field surveys
- **Synthetic Data Generation:** Use GANs to augment training data

### 3. POI and Infrastructure Data

**Sources:**
- **OpenStreetMap:** Comprehensive POI data
  - Overpass API for querying
  - Free and open-source
  
- **Google Places API:** 
  - Rich POI information
  - Real-time data on business hours
  - Cost: $17 per 1000 requests (Places Nearby)

- **Government Databases:**
  - Police station locations
  - Hospital registries
  - Public infrastructure data

### 4. Lighting and Infrastructure

**Sources:**
- **OpenStreetMap:** Street lighting tags (limited coverage)
- **Municipal Corporation Data:** Street light maintenance records
- **Computer Vision Analysis:** Analyze street view images
- **Crowdsourced Data:** User reports on lighting conditions

### 5. Real-time Data

**Sources:**
- **Traffic APIs:** Google Maps Traffic Layer, TomTom Traffic API
- **Weather APIs:** OpenWeatherMap, India Meteorological Department
- **Event Data:** Local event calendars, festival schedules

---

## Development Phases

### Phase 1: MVP (4-6 weeks)
- Basic route calculation with safety scores
- Street lighting analysis using pre-trained CV model
- Simple POI-based crowd density estimation
- Web application with map interface
- 2-3 pilot cities

### Phase 2: Core Features (6-8 weeks)
- Real-time location sharing
- SOS trigger functionality
- Mobile app development
- Historical incident integration
- Enhanced safety score algorithm

### Phase 3: Advanced Features (8-10 weeks)
- Offline map caching
- Community reporting
- ML model fine-tuning
- Multi-language support
- Scalability improvements

### Phase 4: Production (Ongoing)
- Performance optimization
- Security audits
- User feedback integration
- Expansion to more cities
- Partnership with authorities

---

## Success Metrics

### User Metrics
- Daily Active Users (DAU)
- Route completion rate
- SOS trigger response time
- User retention rate (30-day, 90-day)

### Technical Metrics
- Average route calculation time
- Safety score accuracy (validated against ground truth)
- API uptime and response times
- Model inference latency

### Impact Metrics
- User-reported safety incidents (reduction over time)
- User satisfaction scores
- Routes chosen: Safest vs. Fastest ratio
- Emergency response effectiveness

---

## Compliance and Legal

- Privacy policy compliant with IT Act 2000 and DPDP Act 2023
- Terms of service clearly defining liability
- User consent for location tracking and data collection
- Age restriction (18+ or parental consent)
- Disclaimer: App provides recommendations, not guarantees
- Emergency services integration compliance

---

## Budget Considerations (Hackathon Phase)

### Free Tier Options
- OpenStreetMap (unlimited)
- Leaflet.js (open-source)
- Mapillary API (limited free tier)
- Firebase (free tier for small scale)

### Paid Services (Estimate for MVP)
- Google Maps API: $200-500/month (with caching)
- Cloud hosting: $50-100/month (AWS/GCP free tier initially)
- Street View API: $100-200 for initial data collection

### Cost Optimization
- Aggressive caching strategy
- Use OSM as primary, Google as fallback
- Batch API requests
- Implement rate limiting

---

## Risk Mitigation

### Technical Risks
- **API Rate Limits:** Implement caching and fallback options
- **Data Accuracy:** Validate with ground truth, crowdsource corrections
- **Model Bias:** Regular audits, diverse training data

### Operational Risks
- **False Sense of Security:** Clear disclaimers, user education
- **Data Privacy Breach:** Encryption, security audits, minimal data collection
- **Liability Issues:** Legal consultation, proper terms of service

### User Adoption Risks
- **Trust Building:** Transparency in safety score calculation
- **Network Effects:** Incentivize community reporting
- **Competition:** Focus on safety-first USP

---

## Future Enhancements

- AI-powered personal safety assistant (chatbot)
- Integration with wearable devices
- Predictive safety alerts using ML
- Augmented reality navigation
- Integration with ride-sharing services
- Community safety forums
- Safety score API for third-party apps

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Project:** Suraksha-Path - AI for Bharat Hackathon  
**Owner:** [Your Name]
