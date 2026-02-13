# Suraksha-Path - System Design Document

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Safety-Aware Routing Algorithm](#safety-aware-routing-algorithm)
3. [AI/ML Pipeline](#aiml-pipeline)
4. [Database Schema](#database-schema)
5. [API Endpoints](#api-endpoints)
6. [Component Breakdown](#component-breakdown)
7. [Data Flow](#data-flow)
8. [Security Architecture](#security-architecture)
9. [Scalability and Performance](#scalability-and-performance)
10. [Deployment Architecture](#deployment-architecture)

---

## System Architecture

### High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │   Mobile App     │         │   Web App        │                 │
│  │  (React Native)  │         │   (React.js)     │                 │
│  └────────┬─────────┘         └────────┬─────────┘                 │
│           │                             │                            │
│           └─────────────┬───────────────┘                            │
│                         │                                            │
└─────────────────────────┼────────────────────────────────────────────┘
                          │
                          │ HTTPS/REST API
                          │
┌─────────────────────────┼────────────────────────────────────────────┐
│                         │         API GATEWAY                         │
│                    ┌────▼─────┐                                      │
│                    │  Kong /  │                                      │
│                    │  Nginx   │                                      │
│                    └────┬─────┘                                      │
└─────────────────────────┼────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────────┐
│                         │      APPLICATION LAYER                      │
├─────────────────────────┼────────────────────────────────────────────┤
│                         │                                            │
│  ┌──────────────────────▼───────────────────────┐                   │
│  │         FastAPI Backend Server               │                   │
│  │  ┌────────────┐  ┌────────────┐             │                   │
│  │  │   Route    │  │   Safety   │             │                   │
│  │  │  Service   │  │  Service   │             │                   │
│  │  └────────────┘  └────────────┘             │                   │
│  │  ┌────────────┐  ┌────────────┐             │                   │
│  │  │    SOS     │  │  Location  │             │                   │
│  │  │  Service   │  │  Service   │             │                   │
│  │  └────────────┘  └────────────┘             │                   │
│  └──────────────────────┬───────────────────────┘                   │
│                         │                                            │
└─────────────────────────┼────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────────┐
│                         │      ML INFERENCE LAYER                     │
├─────────────────────────┼────────────────────────────────────────────┤
│                         │                                            │
│  ┌──────────────────────▼───────────────────────┐                   │
│  │      ML Inference Engine                     │                   │
│  │  ┌────────────────────────────────────────┐  │                   │
│  │  │  Street Lighting Detection Model       │  │                   │
│  │  │  (EfficientNet/ResNet + Custom Head)   │  │                   │
│  │  └────────────────────────────────────────┘  │                   │
│  │  ┌────────────────────────────────────────┐  │                   │
│  │  │  TensorFlow Serving / TorchServe       │  │                   │
│  │  └────────────────────────────────────────┘  │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────────┐
│                         │       DATA LAYER                            │
├─────────────────────────┼────────────────────────────────────────────┤
│                         │                                            │
│  ┌──────────────────────▼───────────────────────┐                   │
│  │   PostgreSQL + PostGIS (Geospatial DB)      │                   │
│  │   - Users, Incidents, Road Segments         │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                       │
│  ┌──────────────────────────────────────────────┐                   │
│  │   Redis Cache                                │                   │
│  │   - Route Cache, Safety Scores, Sessions    │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                       │
│  ┌──────────────────────────────────────────────┐                   │
│  │   S3 / Object Storage                        │                   │
│  │   - Street View Images, ML Models           │                   │
│  └──────────────────────────────────────────────┘                   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────────────────┐
│                         │    EXTERNAL SERVICES                        │
├─────────────────────────┼────────────────────────────────────────────┤
│                         │                                            │
│  ┌──────────────┐  ┌───▼──────────┐  ┌──────────────┐              │
│  │ Google Maps  │  │ OpenStreetMap│  │   Twilio     │              │
│  │     API      │  │   (OSRM)     │  │   (SMS)      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Mapillary  │  │    Places    │  │   Firebase   │              │
│  │  Street View │  │     API      │  │     FCM      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Architecture Pattern
- **Pattern:** Microservices-oriented with modular monolith for MVP
- **Communication:** RESTful APIs with JSON payloads
- **Authentication:** JWT-based token authentication
- **Caching Strategy:** Multi-level (CDN, Redis, Application-level)

---

## Safety-Aware Routing Algorithm

### Modified Dijkstra's Algorithm

Traditional routing algorithms optimize for shortest distance or time. Suraksha-Path uses a modified Dijkstra's algorithm that optimizes for safety while considering distance.

### Core Concept

**Traditional Dijkstra Weight:**
```
W = distance
```

**Suraksha-Path Safety-Aware Weight:**
```
W = distance × (1 / Safety_Score)
```

Where:
- `distance` is in meters
- `Safety_Score` is normalized to range [0.1, 1.0] (avoiding division by zero)
- Lower weight = better path (safer and/or shorter)

### Mathematical Formulation


#### Step 1: Safety Score Normalization

Raw safety score (0-100) is normalized to [0.1, 1.0]:

```
Safety_Score_Normalized = max(0.1, Raw_Safety_Score / 100)
```

The minimum of 0.1 prevents division by zero and ensures extremely unsafe routes have very high weights (10x penalty).

#### Step 2: Edge Weight Calculation

For each road segment (edge) in the graph:

```
W(u, v) = distance(u, v) × (1 / Safety_Score_Normalized(u, v))
```

**Example:**
- Segment A: 1000m, Safety Score = 80
  - W = 1000 × (1 / 0.8) = 1250
  
- Segment B: 1200m, Safety Score = 95
  - W = 1200 × (1 / 0.95) = 1263.16
  
- Segment C: 800m, Safety Score = 30
  - W = 800 × (1 / 0.3) = 2666.67

Despite being shortest, Segment C has highest weight due to low safety.

#### Step 3: Multi-Objective Optimization

To provide both "Fastest" and "Safest" routes:

**Fastest Route (Traditional):**
```
W_fastest = distance × time_factor
```

**Safest Route (Safety-Optimized):**
```
W_safest = distance × (1 / Safety_Score_Normalized)
```

**Balanced Route (Optional):**
```
W_balanced = α × distance + β × (distance / Safety_Score_Normalized)
```
Where α + β = 1, typically α = 0.5, β = 0.5

### Algorithm Pseudocode

```python
def safety_aware_dijkstra(graph, source, destination, mode='safest'):
    """
    Modified Dijkstra's algorithm for safety-aware routing
    
    Args:
        graph: Road network graph with nodes and edges
        source: Starting location (lat, lon)
        destination: End location (lat, lon)
        mode: 'fastest', 'safest', or 'balanced'
    
    Returns:
        path: List of nodes representing the route
        total_distance: Total distance in meters
        total_safety_score: Aggregate safety score
    """
    
    # Initialize distances and visited set
    distances = {node: float('infinity') for node in graph.nodes}
    distances[source] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited = set(graph.nodes)
    
    # Priority queue: (weight, node)
    pq = [(0, source)]
    
    while pq and destination in unvisited:
        current_weight, current_node = heappop(pq)
        
        if current_node not in unvisited:
            continue
            
        unvisited.remove(current_node)
        
        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor in unvisited:
                edge_data = graph.get_edge_data(current_node, neighbor)
                
                # Calculate weight based on mode
                if mode == 'fastest':
                    weight = edge_data['distance']
                elif mode == 'safest':
                    safety_score = max(0.1, edge_data['safety_score'] / 100)
                    weight = edge_data['distance'] * (1 / safety_score)
                else:  # balanced
                    safety_score = max(0.1, edge_data['safety_score'] / 100)
                    weight = 0.5 * edge_data['distance'] + \
                             0.5 * (edge_data['distance'] / safety_score)
                
                new_distance = distances[current_node] + weight
                
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    heappush(pq, (new_distance, neighbor))
    
    # Reconstruct path
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    path.reverse()
    
    # Calculate metrics
    total_distance = sum(graph.get_edge_data(path[i], path[i+1])['distance'] 
                        for i in range(len(path)-1))
    
    safety_scores = [graph.get_edge_data(path[i], path[i+1])['safety_score'] 
                     for i in range(len(path)-1)]
    total_safety_score = sum(safety_scores) / len(safety_scores)
    
    return path, total_distance, total_safety_score
```

### Time Complexity Analysis

- **Standard Dijkstra:** O((V + E) log V) using binary heap
- **Our Implementation:** O((V + E) log V) - same complexity
- **Space Complexity:** O(V) for distance and previous node storage

Where:
- V = number of intersections/nodes
- E = number of road segments/edges

### Optimization Techniques

1. **A* Enhancement:** Add heuristic for faster convergence
   ```
   f(n) = g(n) + h(n)
   where h(n) = haversine_distance(n, destination) / max_safety_score
   ```

2. **Bidirectional Search:** Search from both source and destination

3. **Graph Preprocessing:** 
   - Contraction Hierarchies for faster queries
   - Pre-compute safety scores for static segments

4. **Spatial Indexing:** R-tree for efficient nearest node lookup

---

## AI/ML Pipeline

### Overview

The ML pipeline processes street-level imagery and geospatial data to generate safety scores for road segments.


### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION PHASE                             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │  Street View Image Collection            │
        │  - Google Street View API                │
        │  - Mapillary API                         │
        │  - Custom field collection               │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  POI Data Collection                     │
        │  - Google Places API                     │
        │  - OpenStreetMap Overpass API            │
        │  - Government databases                  │
        └──────────────┬───────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────┐
│                      │     PREPROCESSING PHASE                       │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Image Preprocessing                     │
        │  - Resize to 224x224 or 384x384          │
        │  - Normalization (ImageNet stats)        │
        │  - Augmentation (training only)          │
        │  - Quality filtering                     │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Geospatial Data Processing              │
        │  - Map POIs to road segments             │
        │  - Calculate density metrics             │
        │  - Time-based weighting                  │
        └──────────────┬───────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────┐
│                      │     ML INFERENCE PHASE                        │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Street Lighting Detection Model         │
        │  (Computer Vision)                       │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Commercial Activity Scoring             │
        │  (Rule-based + ML)                       │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Safety Score Aggregation                │
        │  (Weighted combination)                  │
        └──────────────┬───────────────────────────┘
                       │
┌──────────────────────┼──────────────────────────────────────────────┐
│                      │     STORAGE & SERVING                         │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Store in PostGIS Database               │
        │  - Road segment safety scores            │
        │  - Metadata and timestamps               │
        └──────────────┬───────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────┐
        │  Cache in Redis                          │
        │  - Frequently accessed scores            │
        │  - TTL: 24 hours                         │
        └──────────────────────────────────────────┘
```

### 1. Street Lighting Detection (Computer Vision)

#### Model Architecture

**Base Model:** EfficientNet-B0 or ResNet50 (pre-trained on ImageNet)

**Custom Head:**
```python
class StreetLightingDetector(nn.Module):
    def __init__(self, backbone='efficientnet_b0', num_classes=4):
        super().__init__()
        
        # Load pre-trained backbone
        if backbone == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', 
                                             pretrained=True, 
                                             num_classes=0)
            feature_dim = 1280
        else:  # resnet50
            self.backbone = timm.create_model('resnet50', 
                                             pretrained=True, 
                                             num_classes=0)
            feature_dim = 2048
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
```

**Output Classes:**
1. Well-lit (Score: 90-100)
2. Moderately lit (Score: 60-89)
3. Poorly lit (Score: 30-59)
4. Dark/No lighting (Score: 0-29)

#### Training Process

**Dataset Preparation:**
```python
# Data augmentation for training
train_transforms = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Validation transforms
val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

**Training Configuration:**
- Loss Function: CrossEntropyLoss with class weights
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Batch Size: 32
- Epochs: 50 with early stopping
- Validation Split: 80-20


#### Inference Pipeline

```python
class LightingInferenceService:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        self.transform = self.get_transform()
        
    def load_model(self, path):
        model = StreetLightingDetector()
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        return model
    
    def get_transform(self):
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def predict(self, image_path):
        """
        Predict lighting condition from street view image
        
        Returns:
            lighting_score: 0-100
            confidence: 0-1
            class_name: str
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
        
        # Map class to score
        class_to_score = {
            0: 95,   # Well-lit
            1: 75,   # Moderately lit
            2: 45,   # Poorly lit
            3: 15    # Dark
        }
        
        class_names = ['Well-lit', 'Moderately lit', 'Poorly lit', 'Dark']
        
        lighting_score = class_to_score[predicted_class.item()]
        
        return {
            'lighting_score': lighting_score,
            'confidence': confidence.item(),
            'class': class_names[predicted_class.

#### Inference Pipeline

```python
class LightingInferenceService:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.model.eval()
        self.transform = self.get_transform()
        
    def predict(self, image_path):
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
        
        # Map class to score (0: Well-lit=95, 1: Moderate=75, 2: Poor=45, 3: Dark=15)
        class_to_score = {0: 95, 1: 75, 2: 45, 3: 15}
        lighting_score = class_to_score[predicted_class.item()]
        
        return lighting_score, confidence.item()
```

**Batch Processing for Road Segments:**
```python
def process_road_segment(segment_id, coordinates):
    # Sample 3-5 points along the segment
    sample_points = interpolate_points(coordinates, num_samples=4)
    
    lighting_scores = []
    for point in sample_points:
        # Fetch street view image
        image_url = fetch_street_view(point.lat, point.lon)
        
        # Run inference
        score, confidence = lighting_service.predict(image_url)
        
        if confidence > 0.7:  # Only use high-confidence predictions
            lighting_scores.append(score)
    
    # Average lighting score for segment
    avg_lighting_score = np.mean(lighting_scores) if lighting_scores else 50
    
    return avg_lighting_score
```

### 2. Commercial Activity Scoring (POI Analysis)

#### Data Collection

```python
def fetch_pois_for_segment(segment_coords, radius=100):
    """
    Fetch Points of Interest near road segment
    
    Args:
        segment_coords: List of (lat, lon) tuples
        radius: Search radius in meters
    
    Returns:
        pois: List of POI objects with type and metadata
    """
    center_point = calculate_centroid(segment_coords)
    
    # Query OpenStreetMap
    osm_query = f"""
    [out:json];
    (
      node(around:{radius},{center_point.lat},{center_point.lon})
        ["amenity"];
      way(around:{radius},{center_point.lat},{center_point.lon})
        ["amenity"];
    );
    out body;
    """
    
    osm_pois = query_overpass_api(osm_query)
    
    # Query Google Places (if budget allows)
    google_pois = query_google_places(
        location=center_point,
        radius=radius,
        types=['police', 'hospital', 'restaurant', 'store']
    )
    
    # Merge and deduplicate
    all_pois = merge_pois(osm_pois, google_pois)
    
    return all_pois
```

#### Scoring Logic

```python
class CommercialActivityScorer:
    # POI category weights
    POI_WEIGHTS = {
        'police': 10,
        'hospital': 8,
        'fire_station': 8,
        'pharmacy': 6,
        'restaurant': 5,
        'cafe': 5,
        'convenience_store': 5,
        'bank': 4,
        'atm': 3,
        'fuel_station': 4,
        'hotel': 3,
        'shop': 3,
        'park': 2,
        'default': 1
    }
    
    # Time-based activity multipliers
    TIME_MULTIPLIERS = {
        (6, 10): 1.0,   # Morning
        (10, 18): 1.0,  # Day
        (18, 22): 0.8,  # Evening
        (22, 24): 0.5,  # Late night
        (0, 6): 0.3     # Night
    }
    
    def calculate_score(self, pois, current_hour):
        """Calculate commercial activity score"""
        
        # Get time multiplier
        time_mult = self.get_time_multiplier(current_hour)
        
        # Calculate weighted POI score
        poi_score = 0
        for poi in pois:
            category = poi.get('category', 'default')
            weight = self.POI_WEIGHTS.get(category, 1)
            
            # Distance decay (closer POIs have more impact)
            distance = poi.get('distance', 50)  # meters
            distance_factor = max(0.1, 1 - (distance / 200))
            
            poi_score += weight * distance_factor
        
        # Normalize to 0-100 scale
        normalized_score = min(100, poi_score * 5)
        
        # Apply time multiplier
        final_score = normalized_score * time_mult
        
        return final_score
    
    def get_time_multiplier(self, hour):
        for (start, end), mult in self.TIME_MULTIPLIERS.items():
            if start <= hour < end:
                return mult
        return 0.3
```

### 3. Safety Score Aggregation

```python
class SafetyScoreAggregator:
    def __init__(self):
        self.weights = {
            'lighting': 0.30,
            'commercial_activity': 0.25,
            'historical_incidents': 0.25,
            'road_infrastructure': 0.10,
            'safe_space_proximity': 0.10
        }
    
    def calculate_aggregate_score(self, segment_data):
        """
        Aggregate multiple safety factors into final score
        
        Args:
            segment_data: Dict with all safety factor scores
        
        Returns:
            final_score: 0-100
            breakdown: Dict with individual scores
        """
        scores = {
            'lighting': segment_data.get('lighting_score', 50),
            'commercial_activity': segment_data.get('commercial_score', 50),
            'historical_incidents': self.calculate_incident_score(
                segment_data.get('incidents', [])
            ),
            'road_infrastructure': self.calculate_infrastructure_score(
                segment_data.get('road_type', 'residential')
            ),
            'safe_space_proximity': self.calculate_proximity_score(
                segment_data.get('nearest_safe_space', 500)
            )
        }
        
        # Weighted sum
        final_score = sum(
            scores[factor] * self.weights[factor]
            for factor in self.weights
        )
        
        return round(final_score, 2), scores
    
    def calculate_incident_score(self, incidents):
        """Score based on historical incidents with time decay"""
        if not incidents:
            return 100
        
        current_time = datetime.now()
        weighted_incidents = 0
        
        for incident in incidents:
            # Time decay: incidents lose relevance over time
            days_ago = (current_time - incident['timestamp']).days
            time_factor = max(0.1, 1 - (days_ago / 180))  # 6-month decay
            
            # Severity weight
            severity_weights = {'critical': 10, 'moderate': 5, 'minor': 2}
            severity = severity_weights.get(incident['severity'], 2)
            
            weighted_incidents += severity * time_factor
        
        # Convert to 0-100 score (more incidents = lower score)
        score = max(0, 100 - (weighted_incidents * 5))
        return score
    
    def calculate_infrastructure_score(self, road_type):
        """Score based on road infrastructure"""
        road_scores = {
            'motorway': 85,
            'trunk': 80,
            'primary': 75,
            'secondary': 70,
            'tertiary': 65,
            'residential': 60,
            'service': 50,
            'footway': 45,
            'path': 40
        }
        return road_scores.get(road_type, 60)
    
    def calculate_proximity_score(self, distance_meters):
        """Score based on distance to nearest safe space"""
        # Closer = higher score
        if distance_meters <= 100:
            return 100
        elif distance_meters <= 300:
            return 80
        elif distance_meters <= 500:
            return 60
        elif distance_meters <= 1000:
            return 40
        else:
            return 20
```

---

## Database Schema

### Entity Relationship Diagram

```
┌─────────────────┐         ┌──────────────────┐
│     Users       │         │  EmergencyContacts│
├─────────────────┤         ├──────────────────┤
│ user_id (PK)    │────────<│ contact_id (PK)  │
│ email           │    1:N  │ user_id (FK)     │
│ phone_number    │         │ name             │
│ password_hash   │         │ phone_number     │
│ created_at      │         │ relationship     │
│ last_login      │         └──────────────────┘
│ preferences     │
└────────┬────────┘
         │
         │ 1:N
         │
         ▼
┌─────────────────────────┐
│  ReportedIncidents      │
├─────────────────────────┤
│ incident_id (PK)        │
│ user_id (FK)            │
│ location (GEOGRAPHY)    │
│ incident_type           │
│ severity                │
│ description             │
│ timestamp               │
│ verified                │
│ verification_count      │
└─────────────────────────┘

┌──────────────────────────┐
│  RoadSegments            │
├──────────────────────────┤
│ segment_id (PK)          │
│ geometry (LINESTRING)    │
│ start_node_id            │
│ end_node_id              │
│ distance_meters          │
│ road_type                │
│ road_name                │
│ city                     │
│ safety_score             │
│ lighting_score           │
│ commercial_score         │
│ incident_score           │
│ last_updated             │
└────────┬─────────────────┘
         │
         │ N:M
         │
         ▼
┌──────────────────────────┐
│  Landmarks               │
├──────────────────────────┤
│ landmark_id (PK)         │
│ name                     │
│ location (GEOGRAPHY)     │
│ landmark_type            │
│ category                 │
│ operating_hours          │
│ is_24_7                  │
│ verified                 │
└──────────────────────────┘

┌──────────────────────────┐
│  SafetyScoreHistory      │
├──────────────────────────┤
│ history_id (PK)          │
│ segment_id (FK)          │
│ safety_score             │
│ timestamp                │
│ factors (JSONB)          │
└──────────────────────────┘

┌──────────────────────────┐
│  LocationShares          │
├──────────────────────────┤
│ share_id (PK)            │
│ user_id (FK)             │
│ share_token              │
│ current_location (POINT) │
│ destination (POINT)      │
│ route_geometry (LINE)    │
│ started_at               │
│ expires_at               │
│ is_active                │
│ last_updated             │
└──────────────────────────┘

┌──────────────────────────┐
│  SOSAlerts               │
├──────────────────────────┤
│ alert_id (PK)            │
│ user_id (FK)             │
│ location (GEOGRAPHY)     │
│ triggered_at             │
│ resolved_at              │
│ status                   │
│ response_time_seconds    │
└──────────────────────────┘
```


### Detailed Schema Definitions

#### 1. Users Table

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    phone_number VARCHAR(20) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    date_of_birth DATE,
    gender VARCHAR(20),
    profile_image_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    preferences JSONB DEFAULT '{}',
    
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_phone ON users(phone_number);
CREATE INDEX idx_users_created_at ON users(created_at);
```

**Preferences JSONB Structure:**
```json
{
  "language": "en",
  "theme": "light",
  "notifications_enabled": true,
  "share_location_default": false,
  "preferred_route_type": "balanced",
  "max_detour_minutes": 10
}
```

#### 2. Emergency Contacts Table

```sql
CREATE TABLE emergency_contacts (
    contact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    email VARCHAR(255),
    relationship VARCHAR(100),
    priority_order INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_priority CHECK (priority_order BETWEEN 1 AND 10)
);

CREATE INDEX idx_emergency_contacts_user ON emergency_contacts(user_id);
CREATE INDEX idx_emergency_contacts_priority ON emergency_contacts(user_id, priority_order);
```

#### 3. Reported Incidents Table

```sql
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE reported_incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    incident_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    verified BOOLEAN DEFAULT FALSE,
    verification_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_severity CHECK (severity IN ('minor', 'moderate', 'critical')),
    CONSTRAINT valid_incident_type CHECK (incident_type IN 
        ('theft', 'assault', 'harassment', 'suspicious_activity', 'other')),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'verified', 'rejected', 'resolved'))
);

CREATE INDEX idx_incidents_location ON reported_incidents USING GIST(location);
CREATE INDEX idx_incidents_timestamp ON reported_incidents(timestamp DESC);
CREATE INDEX idx_incidents_type ON reported_incidents(incident_type);
CREATE INDEX idx_incidents_severity ON reported_incidents(severity);
CREATE INDEX idx_incidents_verified ON reported_incidents(verified) WHERE verified = TRUE;
```

#### 4. Road Segments Table

```sql
CREATE TABLE road_segments (
    segment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    osm_way_id BIGINT UNIQUE,
    geometry GEOGRAPHY(LINESTRING, 4326) NOT NULL,
    start_node_id BIGINT,
    end_node_id BIGINT,
    distance_meters DECIMAL(10, 2) NOT NULL,
    road_type VARCHAR(50),
    road_name VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    country VARCHAR(100) DEFAULT 'India',
    
    -- Safety scores
    safety_score DECIMAL(5, 2) DEFAULT 50.0,
    lighting_score DECIMAL(5, 2),
    commercial_score DECIMAL(5, 2),
    incident_score DECIMAL(5, 2),
    infrastructure_score DECIMAL(5, 2),
    proximity_score DECIMAL(5, 2),
    
    -- Metadata
    max_speed_kmh INTEGER,
    lanes INTEGER,
    surface_type VARCHAR(50),
    has_sidewalk BOOLEAN,
    has_cctv BOOLEAN DEFAULT FALSE,
    is_one_way BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_safety_score CHECK (safety_score BETWEEN 0 AND 100),
    CONSTRAINT valid_distance CHECK (distance_meters > 0)
);

CREATE INDEX idx_segments_geometry ON road_segments USING GIST(geometry);
CREATE INDEX idx_segments_city ON road_segments(city);
CREATE INDEX idx_segments_safety_score ON road_segments(safety_score);
CREATE INDEX idx_segments_road_type ON road_segments(road_type);
CREATE INDEX idx_segments_osm_way ON road_segments(osm_way_id);

-- Spatial index for fast nearest segment queries
CREATE INDEX idx_segments_geography ON road_segments USING GIST(geography(geometry));
```

#### 5. Landmarks Table

```sql
CREATE TABLE landmarks (
    landmark_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    osm_id BIGINT,
    google_place_id VARCHAR(255),
    name VARCHAR(255) NOT NULL,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    landmark_type VARCHAR(50) NOT NULL,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    address TEXT,
    phone_number VARCHAR(20),
    operating_hours JSONB,
    is_24_7 BOOLEAN DEFAULT FALSE,
    verified BOOLEAN DEFAULT FALSE,
    rating DECIMAL(3, 2),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT valid_landmark_type CHECK (landmark_type IN 
        ('police_station', 'hospital', 'fire_station', 'pharmacy', 
         'restaurant', 'cafe', 'store', 'bank', 'atm', 'hotel', 'other')),
    CONSTRAINT valid_rating CHECK (rating BETWEEN 0 AND 5)
);

CREATE INDEX idx_landmarks_location ON landmarks USING GIST(location);
CREATE INDEX idx_landmarks_type ON landmarks(landmark_type);
CREATE INDEX idx_landmarks_category ON landmarks(category);
CREATE INDEX idx_landmarks_24_7 ON landmarks(is_24_7) WHERE is_24_7 = TRUE;
CREATE INDEX idx_landmarks_verified ON landmarks(verified) WHERE verified = TRUE;
```

**Operating Hours JSONB Structure:**
```json
{
  "monday": {"open": "09:00", "close": "21:00"},
  "tuesday": {"open": "09:00", "close": "21:00"},
  "wednesday": {"open": "09:00", "close": "21:00"},
  "thursday": {"open": "09:00", "close": "21:00"},
  "friday": {"open": "09:00", "close": "22:00"},
  "saturday": {"open": "10:00", "close": "22:00"},
  "sunday": {"open": "10:00", "close": "20:00"}
}
```

#### 6. Safety Score History Table

```sql
CREATE TABLE safety_score_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    segment_id UUID NOT NULL REFERENCES road_segments(segment_id) ON DELETE CASCADE,
    safety_score DECIMAL(5, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    factors JSONB NOT NULL,
    calculation_version VARCHAR(20),
    
    CONSTRAINT valid_safety_score_hist CHECK (safety_score BETWEEN 0 AND 100)
);

CREATE INDEX idx_score_history_segment ON safety_score_history(segment_id);
CREATE INDEX idx_score_history_timestamp ON safety_score_history(timestamp DESC);
CREATE INDEX idx_score_history_segment_time ON safety_score_history(segment_id, timestamp DESC);
```

**Factors JSONB Structure:**
```json
{
  "lighting_score": 85.5,
  "commercial_score": 72.3,
  "incident_score": 90.0,
  "infrastructure_score": 75.0,
  "proximity_score": 80.0,
  "time_of_day": "20:30",
  "weather_condition": "clear"
}
```

#### 7. Location Shares Table

```sql
CREATE TABLE location_shares (
    share_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    share_token VARCHAR(64) UNIQUE NOT NULL,
    current_location GEOGRAPHY(POINT, 4326),
    destination GEOGRAPHY(POINT, 4326),
    route_geometry GEOGRAPHY(LINESTRING, 4326),
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shared_with JSONB DEFAULT '[]',
    
    CONSTRAINT valid_expiry CHECK (expires_at > started_at)
);

CREATE INDEX idx_location_shares_user ON location_shares(user_id);
CREATE INDEX idx_location_shares_token ON location_shares(share_token);
CREATE INDEX idx_location_shares_active ON location_shares(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_location_shares_expires ON location_shares(expires_at);
```

#### 8. SOS Alerts Table

```sql
CREATE TABLE sos_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    triggered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',
    response_time_seconds INTEGER,
    alert_type VARCHAR(50) DEFAULT 'manual',
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT valid_status CHECK (status IN ('active', 'resolved', 'false_alarm', 'cancelled')),
    CONSTRAINT valid_alert_type CHECK (alert_type IN ('manual', 'shake', 'automated'))
);

CREATE INDEX idx_sos_alerts_user ON sos_alerts(user_id);
CREATE INDEX idx_sos_alerts_location ON sos_alerts USING GIST(location);
CREATE INDEX idx_sos_alerts_triggered ON sos_alerts(triggered_at DESC);
CREATE INDEX idx_sos_alerts_status ON sos_alerts(status);
CREATE INDEX idx_sos_alerts_active ON sos_alerts(status) WHERE status = 'active';
```

### Database Functions and Triggers

#### Auto-update timestamp trigger

```sql
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_landmarks_updated_at
    BEFORE UPDATE ON landmarks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

#### Geospatial query functions

```sql
-- Find nearest road segment to a point
CREATE OR REPLACE FUNCTION find_nearest_segment(
    lat DECIMAL,
    lon DECIMAL,
    max_distance_meters INTEGER DEFAULT 100
)
RETURNS TABLE (
    segment_id UUID,
    distance_meters DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rs.segment_id,
        ST_Distance(
            rs.geometry::geography,
            ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography
        ) as distance_meters
    FROM road_segments rs
    WHERE ST_DWithin(
        rs.geometry::geography,
        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography,
        max_distance_meters
    )
    ORDER BY distance_meters
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Find incidents near a road segment
CREATE OR REPLACE FUNCTION get_incidents_near_segment(
    seg_id UUID,
    radius_meters INTEGER DEFAULT 200,
    days_back INTEGER DEFAULT 180
)
RETURNS TABLE (
    incident_id UUID,
    incident_type VARCHAR,
    severity VARCHAR,
    distance_meters DECIMAL,
    days_ago INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ri.incident_id,
        ri.incident_type,
        ri.severity,
        ST_Distance(
            ri.location::geography,
            rs.geometry::geography
        ) as distance_meters,
        EXTRACT(DAY FROM CURRENT_TIMESTAMP - ri.timestamp)::INTEGER as days_ago
    FROM reported_incidents ri
    CROSS JOIN road_segments rs
    WHERE rs.segment_id = seg_id
        AND ri.verified = TRUE
        AND ri.timestamp > CURRENT_TIMESTAMP - INTERVAL '1 day' * days_back
        AND ST_DWithin(
            ri.location::geography,
            rs.geometry::geography,
            radius_meters
        )
    ORDER BY distance_meters;
END;
$$ LANGUAGE plpgsql;
```

---

## API Endpoints

### Base URL
```
Production: https://api.suraksha-path.in/v1
Development: http://localhost:8000/v1
```

### Authentication
All protected endpoints require JWT token in header:
```
Authorization: Bearer <jwt_token>
```


### 1. Route Endpoints

#### GET /get-safe-route

Get optimized route with safety considerations.

**Request:**
```json
{
  "origin": {
    "lat": 28.6139,
    "lon": 77.2090
  },
  "destination": {
    "lat": 28.5355,
    "lon": 77.3910
  },
  "mode": "safest",  // "fastest", "safest", "balanced"
  "time_of_travel": "2024-03-15T20:30:00Z",  // Optional, defaults to now
  "avoid_areas": [  // Optional
    {
      "lat": 28.5800,
      "lon": 77.3200,
      "radius_meters": 500
    }
  ],
  "preferences": {
    "max_detour_minutes": 15,
    "min_safety_score": 60
  }
}
```

**Response:**
```json
{
  "status": "success",
  "routes": [
    {
      "route_id": "route_safest_abc123",
      "type": "safest",
      "geometry": {
        "type": "LineString",
        "coordinates": [[77.2090, 28.6139], [77.2100, 28.6140], ...]
      },
      "summary": {
        "total_distance_meters": 25430,
        "estimated_time_seconds": 1820,
        "average_safety_score": 82.5,
        "total_segments": 45
      },
      "segments": [
        {
          "segment_id": "seg_001",
          "distance_meters": 450,
          "safety_score": 85.2,
          "road_name": "MG Road",
          "road_type": "primary",
          "safety_factors": {
            "lighting": 90,
            "commercial_activity": 85,
            "incidents": 80,
            "infrastructure": 85,
            "proximity_to_safe_spaces": 75
          }
        }
      ],
      "waypoints": [
        {
          "lat": 28.6139,
          "lon": 77.2090,
          "instruction": "Head north on MG Road"
        }
      ],
      "safety_highlights": [
        "Well-lit streets throughout",
        "3 police stations within 500m",
        "High commercial activity area"
      ],
      "safety_concerns": [
        "Low activity between 11 PM - 5 AM on Sector 18 Road"
      ]
    },
    {
      "route_id": "route_fastest_xyz789",
      "type": "fastest",
      "geometry": {...},
      "summary": {
        "total_distance_meters": 22100,
        "estimated_time_seconds": 1320,
        "average_safety_score": 65.3,
        "total_segments": 38
      },
      "comparison_with_safest": {
        "distance_difference_meters": -3330,
        "time_difference_seconds": -500,
        "safety_score_difference": -17.2
      }
    }
  ],
  "metadata": {
    "calculation_time_ms": 245,
    "timestamp": "2024-03-15T20:30:15Z",
    "api_version": "1.0"
  }
}
```

**Status Codes:**
- 200: Success
- 400: Invalid request parameters
- 404: No route found
- 429: Rate limit exceeded
- 500: Server error

---

#### POST /report-incident

Report a safety incident.

**Request:**
```json
{
  "location": {
    "lat": 28.6139,
    "lon": 77.2090
  },
  "incident_type": "harassment",  // theft, assault, harassment, suspicious_activity, other
  "severity": "moderate",  // minor, moderate, critical
  "description": "Suspicious individual following pedestrians",
  "timestamp": "2024-03-15T20:15:00Z",  // Optional, defaults to now
  "is_anonymous": false,
  "media_urls": [  // Optional
    "https://storage.suraksha-path.in/incidents/img_001.jpg"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "incident_id": "inc_abc123xyz",
  "message": "Incident reported successfully. It will be verified by our team.",
  "verification_status": "pending",
  "estimated_verification_time_hours": 2,
  "nearby_safe_spaces": [
    {
      "name": "Connaught Place Police Station",
      "type": "police_station",
      "distance_meters": 450,
      "phone": "+91-11-23412345"
    }
  ]
}
```

**Status Codes:**
- 201: Incident created
- 400: Invalid data
- 401: Unauthorized (if not anonymous)
- 429: Rate limit exceeded

---

#### GET /get-safety-score

Get safety score for a specific location or route segment.

**Request:**
```
GET /get-safety-score?lat=28.6139&lon=77.2090&radius=100&time=2024-03-15T20:30:00Z
```

**Query Parameters:**
- `lat` (required): Latitude
- `lon` (required): Longitude
- `radius` (optional): Radius in meters (default: 100)
- `time` (optional): Time for score calculation (default: now)

**Response:**
```json
{
  "status": "success",
  "location": {
    "lat": 28.6139,
    "lon": 77.2090
  },
  "safety_score": 82.5,
  "score_breakdown": {
    "lighting": {
      "score": 90,
      "weight": 0.30,
      "contribution": 27.0,
      "description": "Well-lit area with adequate street lighting"
    },
    "commercial_activity": {
      "score": 85,
      "weight": 0.25,
      "contribution": 21.25,
      "description": "High commercial activity with 15 active businesses nearby"
    },
    "historical_incidents": {
      "score": 75,
      "weight": 0.25,
      "contribution": 18.75,
      "description": "2 minor incidents reported in last 6 months"
    },
    "infrastructure": {
      "score": 80,
      "weight": 0.10,
      "contribution": 8.0,
      "description": "Primary road with good infrastructure"
    },
    "safe_space_proximity": {
      "score": 75,
      "weight": 0.10,
      "contribution": 7.5,
      "description": "Police station 450m away"
    }
  },
  "nearby_segments": [
    {
      "segment_id": "seg_001",
      "road_name": "MG Road",
      "safety_score": 85.2,
      "distance_meters": 25
    }
  ],
  "nearby_incidents": [
    {
      "incident_type": "theft",
      "severity": "minor",
      "distance_meters": 180,
      "days_ago": 45
    }
  ],
  "nearby_safe_spaces": [
    {
      "name": "Connaught Place Police Station",
      "type": "police_station",
      "distance_meters": 450,
      "is_24_7": true
    }
  ],
  "recommendations": [
    "This area is generally safe during evening hours",
    "Stay on main roads for optimal safety"
  ],
  "timestamp": "2024-03-15T20:30:00Z"
}
```

---

### 2. Location Sharing Endpoints

#### POST /location/share

Start sharing live location.

**Request:**
```json
{
  "destination": {
    "lat": 28.5355,
    "lon": 77.3910
  },
  "route_geometry": {
    "type": "LineString",
    "coordinates": [[77.2090, 28.6139], ...]
  },
  "duration_minutes": 60,
  "share_with": [
    {
      "contact_id": "contact_123",
      "notification_method": "sms"  // sms, whatsapp, email
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "share_id": "share_abc123",
  "share_token": "sk_live_xyz789abc",
  "share_url": "https://suraksha-path.in/track/sk_live_xyz789abc",
  "expires_at": "2024-03-15T21:30:00Z",
  "notifications_sent": 2
}
```

---

#### PUT /location/share/{share_id}

Update current location during active sharing.

**Request:**
```json
{
  "current_location": {
    "lat": 28.6200,
    "lon": 77.2150
  },
  "speed_kmh": 45,
  "bearing_degrees": 135
}
```

**Response:**
```json
{
  "status": "success",
  "updated_at": "2024-03-15T20:35:00Z",
  "eta_seconds": 1200,
  "distance_remaining_meters": 18500
}
```

---

#### POST /location/share/{share_id}/complete

Mark journey as complete.

**Request:**
```json
{
  "status": "reached_safely"  // reached_safely, cancelled
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Journey completed. Notifications sent to contacts.",
  "journey_summary": {
    "duration_minutes": 32,
    "distance_meters": 25430,
    "average_speed_kmh": 47.6
  }
}
```

---

### 3. SOS Endpoints

#### POST /sos/trigger

Trigger SOS alert.

**Request:**
```json
{
  "location": {
    "lat": 28.6139,
    "lon": 77.2090
  },
  "alert_type": "manual",  // manual, shake, automated
  "message": "Need immediate help"  // Optional
}
```

**Response:**
```json
{
  "status": "success",
  "alert_id": "sos_abc123",
  "message": "SOS alert triggered. Emergency contacts notified.",
  "notifications_sent": {
    "emergency_contacts": 3,
    "local_authorities": 1
  },
  "nearest_help": [
    {
      "name": "Connaught Place Police Station",
      "type": "police_station",
      "distance_meters": 450,
      "phone": "+91-11-23412345",
      "eta_minutes": 5
    }
  ],
  "triggered_at": "2024-03-15T20:30:00Z"
}
```

---

### 4. User Management Endpoints

#### POST /auth/register

Register new user.

**Request:**
```json
{
  "email": "user@example.com",
  "phone_number": "+919876543210",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "status": "success",
  "user_id": "user_abc123",
  "message": "Registration successful. Please verify your email.",
  "verification_sent": true
}
```

---

#### POST /auth/login

User login.

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:**
```json
{
  "status": "success",
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "user": {
    "user_id": "user_abc123",
    "email": "user@example.com",
    "full_name": "John Doe"
  }
}
```

---

#### POST /users/emergency-contacts

Add emergency contact.

**Request:**
```json
{
  "name": "Jane Doe",
  "phone_number": "+919876543211",
  "email": "jane@example.com",
  "relationship": "Sister",
  "priority_order": 1
}
```

**Response:**
```json
{
  "status": "success",
  "contact_id": "contact_abc123",
  "message": "Emergency contact added successfully"
}
```

---

### API Rate Limits

| Endpoint Category | Rate Limit | Window |
|------------------|------------|--------|
| Route calculation | 60 requests | 1 minute |
| Safety score | 120 requests | 1 minute |
| Incident reporting | 10 requests | 1 hour |
| Location updates | 120 requests | 1 minute |
| SOS trigger | 5 requests | 1 minute |
| Authentication | 10 requests | 5 minutes |

---

## Component Breakdown


### 1. Frontend Layer

#### Mobile Application (React Native)

**Technology Stack:**
- React Native 0.72+
- TypeScript
- React Navigation 6.x
- Redux Toolkit (state management)
- React Native Maps (Mapbox GL)
- Axios (HTTP client)
- AsyncStorage (local storage)
- React Native Geolocation
- React Native Push Notifications

**Key Components:**

```
src/
├── components/
│   ├── Map/
│   │   ├── SafetyMap.tsx           # Main map component
│   │   ├── RouteOverlay.tsx        # Route visualization
│   │   ├── SafetyHeatmap.tsx       # Safety score heatmap
│   │   └── MarkerCluster.tsx       # Incident markers
│   ├── Route/
│   │   ├── RouteComparison.tsx     # Side-by-side route comparison
│   │   ├── RouteCard.tsx           # Individual route display
│   │   └── SafetyBreakdown.tsx     # Safety score breakdown
│   ├── SOS/
│   │   ├── SOSButton.tsx           # Emergency button
│   │   ├── SOSConfirmation.tsx     # Confirmation dialog
│   │   └── EmergencyContacts.tsx   # Contact list
│   └── Location/
│       ├── LocationShare.tsx       # Location sharing UI
│       └── LiveTracking.tsx        # Real-time tracking
├── screens/
│   ├── HomeScreen.tsx
│   ├── RouteSearchScreen.tsx
│   ├── RouteResultsScreen.tsx
│   ├── NavigationScreen.tsx
│   ├── ProfileScreen.tsx
│   ├── IncidentReportScreen.tsx
│   └── EmergencyScreen.tsx
├── services/
│   ├── api.service.ts              # API client
│   ├── location.service.ts         # Location tracking
│   ├── notification.service.ts     # Push notifications
│   └── storage.service.ts          # Local storage
├── store/
│   ├── slices/
│   │   ├── authSlice.ts
│   │   ├── routeSlice.ts
│   │   ├── locationSlice.ts
│   │   └── sosSlice.ts
│   └── store.ts
└── utils/
    ├── geolocation.ts
    ├── permissions.ts
    └── constants.ts
```

**Key Features Implementation:**

```typescript
// SafetyMap.tsx - Main map component
import React, { useEffect, useState } from 'react';
import MapboxGL from '@rnmapbox/maps';
import { useSelector, useDispatch } from 'react-redux';

const SafetyMap: React.FC = () => {
  const [userLocation, setUserLocation] = useState(null);
  const routes = useSelector(state => state.route.routes);
  const safetyScores = useSelector(state => state.route.safetyScores);
  
  useEffect(() => {
    // Initialize location tracking
    startLocationTracking();
  }, []);
  
  const renderRoutes = () => {
    return routes.map(route => (
      <MapboxGL.ShapeSource
        key={route.route_id}
        id={route.route_id}
        shape={route.geometry}
      >
        <MapboxGL.LineLayer
          id={`${route.route_id}-line`}
          style={{
            lineColor: getRouteColor(route.average_safety_score),
            lineWidth: 6,
            lineOpacity: 0.8
          }}
        />
      </MapboxGL.ShapeSource>
    ));
  };
  
  const getRouteColor = (safetyScore: number) => {
    if (safetyScore >= 80) return '#22c55e'; // Green
    if (safetyScore >= 60) return '#eab308'; // Yellow
    return '#ef4444'; // Red
  };
  
  return (
    <MapboxGL.MapView style={{ flex: 1 }}>
      <MapboxGL.Camera
        zoomLevel={14}
        centerCoordinate={userLocation}
      />
      {renderRoutes()}
      <MapboxGL.UserLocation visible={true} />
    </MapboxGL.MapView>
  );
};
```

---

#### Web Application (React.js)

**Technology Stack:**
- React 18+
- TypeScript
- React Router v6
- Redux Toolkit
- Leaflet / Mapbox GL JS
- Axios
- TailwindCSS / Material-UI
- Chart.js (visualizations)

**Project Structure:**

```
src/
├── components/
│   ├── map/
│   │   ├── InteractiveMap.tsx
│   │   ├── RouteLayer.tsx
│   │   └── SafetyHeatmap.tsx
│   ├── route/
│   │   ├── RouteSearch.tsx
│   │   ├── RouteComparison.tsx
│   │   └── SafetyMetrics.tsx
│   ├── dashboard/
│   │   ├── SafetyDashboard.tsx
│   │   └── IncidentMap.tsx
│   └── common/
│       ├── Header.tsx
│       ├── Footer.tsx
│       └── LoadingSpinner.tsx
├── pages/
│   ├── HomePage.tsx
│   ├── RoutePlannerPage.tsx
│   ├── SafetyScorePage.tsx
│   ├── IncidentReportPage.tsx
│   └── ProfilePage.tsx
├── services/
│   ├── apiClient.ts
│   ├── routeService.ts
│   ├── safetyService.ts
│   └── authService.ts
├── hooks/
│   ├── useGeolocation.ts
│   ├── useRoute.ts
│   └── useSafetyScore.ts
├── store/
│   └── slices/
└── utils/
```

---

### 2. Backend Layer (FastAPI)

**Technology Stack:**
- Python 3.9+
- FastAPI 0.104+
- Uvicorn (ASGI server)
- SQLAlchemy 2.0 (ORM)
- Alembic (migrations)
- Pydantic (validation)
- PyJWT (authentication)
- Redis-py (caching)
- Celery (task queue)
- HTTPX (async HTTP client)

**Project Structure:**

```
backend/
├── app/
│   ├── main.py                     # FastAPI application
│   ├── config.py                   # Configuration
│   ├── dependencies.py             # Dependency injection
│   │
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── routes.py       # Route endpoints
│   │   │   │   ├── safety.py       # Safety score endpoints
│   │   │   │   ├── incidents.py    # Incident reporting
│   │   │   │   ├── location.py     # Location sharing
│   │   │   │   ├── sos.py          # SOS endpoints
│   │   │   │   └── auth.py         # Authentication
│   │   │   └── router.py
│   │   └── deps.py
│   │
│   ├── core/
│   │   ├── security.py             # JWT, password hashing
│   │   ├── config.py               # Settings
│   │   └── logging.py              # Logging configuration
│   │
│   ├── models/
│   │   ├── user.py
│   │   ├── incident.py
│   │   ├── road_segment.py
│   │   ├── landmark.py
│   │   └── location_share.py
│   │
│   ├── schemas/
│   │   ├── route.py                # Pydantic schemas
│   │   ├── safety.py
│   │   ├── incident.py
│   │   └── user.py
│   │
│   ├── services/
│   │   ├── route_service.py        # Route calculation logic
│   │   ├── safety_service.py       # Safety score calculation
│   │   ├── ml_service.py           # ML model inference
│   │   ├── maps_service.py         # External maps API
│   │   ├── notification_service.py # SMS/Push notifications
│   │   └── cache_service.py        # Redis caching
│   │
│   ├── algorithms/
│   │   ├── dijkstra.py             # Modified Dijkstra
│   │   ├── graph_builder.py        # Road network graph
│   │   └── safety_calculator.py    # Safety score aggregation
│   │
│   ├── db/
│   │   ├── session.py              # Database session
│   │   ├── base.py                 # Base model
│   │   └── init_db.py              # Database initialization
│   │
│   └── utils/
│       ├── geospatial.py           # Geospatial utilities
│       ├── validators.py           # Custom validators
│       └── helpers.py
│
├── alembic/                        # Database migrations
├── tests/
├── requirements.txt
└── Dockerfile
```

**Key Service Implementation:**

```python
# services/route_service.py
from typing import List, Dict, Tuple
import networkx as nx
from app.algorithms.dijkstra import safety_aware_dijkstra
from app.services.safety_service import SafetyService
from app.services.cache_service import CacheService

class RouteService:
    def __init__(self, db_session, cache: CacheService):
        self.db = db_session
        self.cache = cache
        self.safety_service = SafetyService(db_session)
        self.graph = None
    
    async def calculate_routes(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        mode: str = 'safest',
        time_of_travel: datetime = None
    ) -> List[Dict]:
        """Calculate multiple route options"""
        
        # Check cache
        cache_key = f"route:{origin}:{destination}:{mode}"
        cached_route = await self.cache.get(cache_key)
        if cached_route:
            return cached_route
        
        # Build road network graph
        if not self.graph:
            self.graph = await self.build_graph(origin, destination)
        
        # Find nearest nodes
        start_node = self.find_nearest_node(origin)
        end_node = self.find_nearest_node(destination)
        
        # Calculate routes
        routes = []
        
        # Safest route
        if mode in ['safest', 'all']:
            safest_path, distance, safety_score = safety_aware_dijkstra(
                self.graph, start_node, end_node, mode='safest'
            )
            routes.append(self.format_route(safest_path, 'safest'))
        
        # Fastest route
        if mode in ['fastest', 'all']:
            fastest_path, distance, safety_score = safety_aware_dijkstra(
                self.graph, start_node, end_node, mode='fastest'
            )
            routes.append(self.format_route(fastest_path, 'fastest'))
        
        # Cache result
        await self.cache.set(cache_key, routes, ttl=3600)
        
        return routes
    
    async def build_graph(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float]
    ) -> nx.DiGraph:
        """Build road network graph with safety scores"""
        
        # Query road segments in bounding box
        segments = await self.get_segments_in_bbox(origin, destination)
        
        # Create graph
        G = nx.DiGraph()
        
        for segment in segments:
            # Add edge with safety-aware weight
            safety_score = await self.safety_service.get_segment_score(
                segment.segment_id
            )
            
            weight = segment.distance_meters * (1 / max(0.1, safety_score / 100))
            
            G.add_edge(
                segment.start_node_id,
                segment.end_node_id,
                distance=segment.distance_meters,
                safety_score=safety_score,
                weight=weight,
                segment_id=segment.segment_id,
                road_name=segment.road_name
            )
        
        return G
```

---

### 3. ML Inference Engine

**Technology Stack:**
- TensorFlow 2.x / PyTorch 1.13+
- TensorFlow Serving / TorchServe
- ONNX Runtime (optimization)
- FastAPI (inference API)
- Docker

**Structure:**

```
ml_service/
├── models/
│   ├── street_lighting/
│   │   ├── model.pth
│   │   ├── config.json
│   │   └── preprocessing.py
│   └── model_registry.py
│
├── inference/
│   ├── lighting_detector.py
│   ├── batch_processor.py
│   └── model_loader.py
│
├── api/
│   ├── main.py                     # FastAPI app
│   └── endpoints.py
│
├── utils/
│   ├── image_processing.py
│   └── postprocessing.py
│
└── Dockerfile
```

**Inference API:**

```python
# ml_service/api/main.py
from fastapi import FastAPI, File, UploadFile
from inference.lighting_detector import LightingDetector

app = FastAPI()
detector = LightingDetector(model_path='models/street_lighting/model.pth')

@app.post("/predict/lighting")
async def predict_lighting(image: UploadFile = File(...)):
    """Predict street lighting condition"""
    
    # Read image
    contents = await image.read()
    
    # Run inference
    result = detector.predict(contents)
    
    return {
        "lighting_score": result['score'],
        "confidence": result['confidence'],
        "class": result['class'],
        "inference_time_ms": result['inference_time']
    }

@app.post("/predict/batch")
async def predict_batch(segment_id: str, coordinates: List[Dict]):
    """Batch prediction for road segment"""
    
    results = await detector.predict_segment(segment_id, coordinates)
    
    return {
        "segment_id": segment_id,
        "average_lighting_score": results['avg_score'],
        "sample_count": len(results['samples']),
        "samples": results['samples']
    }
```

---

### 4. Geospatial Database (PostGIS)

**Configuration:**
- PostgreSQL 15+
- PostGIS 3.3+
- Connection pooling (PgBouncer)
- Read replicas for scaling

**Optimization:**
- Spatial indexes on all geography columns
- Partitioning for large tables (incidents, history)
- Materialized views for common queries
- Query optimization with EXPLAIN ANALYZE

---

### 5. Caching Layer (Redis)

**Use Cases:**
- Route caching (TTL: 1 hour)
- Safety score caching (TTL: 24 hours)
- Session management
- Rate limiting
- Real-time location tracking

**Configuration:**
```python
# Redis key patterns
ROUTE_CACHE = "route:{origin}:{destination}:{mode}"
SAFETY_SCORE = "safety:{segment_id}:{timestamp}"
USER_SESSION = "session:{user_id}"
RATE_LIMIT = "ratelimit:{user_id}:{endpoint}"
LOCATION_SHARE = "location:{share_id}"
```

---

## Data Flow

### Route Calculation Flow

```
1. User Request
   ↓
2. API Gateway (Authentication, Rate Limiting)
   ↓
3. FastAPI Backend
   ↓
4. Check Redis Cache
   ├─ Cache Hit → Return cached route
   └─ Cache Miss ↓
5. Query PostGIS for road segments
   ↓
6. Build road network graph
   ↓
7. For each segment:
   ├─ Check safety score cache
   ├─ If missing: Calculate safety score
   │   ├─ Query ML service for lighting
   │   ├─ Query POI data for commercial activity
   │   ├─ Query incidents from database
   │   └─ Aggregate scores
   └─ Add to graph with weights
   ↓
8. Run modified Dijkstra's algorithm
   ↓
9. Format route response
   ↓
10. Cache result in Redis
   ↓
11. Return to client
```

### Safety Score Calculation Flow

```
1. Segment identified
   ↓
2. Check Redis cache
   ├─ Cache Hit → Return score
   └─ Cache Miss ↓
3. Parallel data collection:
   ├─ Street View images → ML Service
   ├─ POI data → Maps API
   ├─ Incidents → PostGIS query
   └─ Infrastructure → OSM data
   ↓
4. ML Inference (if needed)
   ├─ Fetch street view images
   ├─ Preprocess images
   ├─ Run lighting detection model
   └─ Return lighting scores
   ↓
5. Aggregate all factors
   ↓
6. Calculate weighted safety score
   ↓
7. Store in database
   ↓
8. Cache in Redis (24h TTL)
   ↓
9. Return score
```

---

## Security Architecture

### Authentication & Authorization

**JWT Token Structure:**
```json
{
  "user_id": "user_abc123",
  "email": "user@example.com",
  "roles": ["user"],
  "iat": 1710532800,
  "exp": 1710536400
}
```

**Security Measures:**
- Password hashing: bcrypt (cost factor: 12)
- JWT tokens: HS256 algorithm
- Refresh token rotation
- Rate limiting per user/IP
- HTTPS only in production
- CORS configuration
- SQL injection prevention (parameterized queries)
- XSS protection (input sanitization)

### Data Privacy

- Location data encrypted at rest (AES-256)
- PII anonymization for analytics
- User consent for data collection
- Right to deletion (GDPR compliance)
- Minimal data retention (90 days for location history)

---

## Scalability and Performance

### Horizontal Scaling

**Application Layer:**
- Stateless FastAPI instances
- Load balancer (Nginx/AWS ALB)
- Auto-scaling based on CPU/memory

**Database Layer:**
- Read replicas for queries
- Write master for updates
- Connection pooling

**Caching Layer:**
- Redis Cluster for high availability
- Cache warming for popular routes

### Performance Optimizations

1. **Database:**
   - Spatial indexes
   - Query optimization
   - Materialized views
   - Partitioning

2. **API:**
   - Response compression (gzip)
   - Pagination for large results
   - Field selection (sparse fieldsets)
   - Async I/O

3. **ML Inference:**
   - Model quantization (INT8)
   - Batch processing
   - GPU acceleration
   - Model caching

4. **Frontend:**
   - Code splitting
   - Lazy loading
   - Image optimization
   - Service workers (PWA)

---

## Deployment Architecture

### Production Environment

```
┌─────────────────────────────────────────────────────────────┐
│                     Load Balancer (AWS ALB)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │ FastAPI │    │ FastAPI │    │ FastAPI │
    │Instance1│    │Instance2│    │Instance3│
    └────┬────┘    └────┬────┘    └────┬────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
    │PostgreSQL│    │  Redis  │    │   ML    │
    │ Primary  │    │ Cluster │    │ Service │
    └────┬────┘    └─────────┘    └─────────┘
         │
    ┌────▼────┐
    │PostgreSQL│
    │ Replica  │
    └─────────┘
```

### Container Orchestration (Kubernetes)

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: suraksha-path-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: suraksha-path-api
  template:
    metadata:
      labels:
        app: suraksha-path-api
    spec:
      containers:
      - name: api
        image: suraksha-path/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Project:** Suraksha-Path - System Design  
**Author:** [Your Name]
