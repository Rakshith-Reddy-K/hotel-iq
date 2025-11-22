# HotelIQ Architecture Diagram

## System Flow Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                            FRONTEND                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Hotel Page 1 │  │ Hotel Page 2 │  │ Hotel Page N │              │
│  │  (ID: 1)     │  │  (ID: 2)     │  │  (ID: N)     │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                        │
│         └─────────────────┴─────────────────┘                        │
│                           │                                          │
│                  User Query + Hotel ID                              │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      API ENDPOINT                                    │
│              POST /api/v1/chat/message                              │
│         { message, user_id, hotel_id, thread_id }                   │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CHAT SERVICE                                    │
│  • Creates/retrieves thread                                         │
│  • Manages conversation history                                     │
│  • Initializes state with hotel_id                                  │
└───────────────────────────┼─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT GRAPH WORKFLOW                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    ENTRY POINT                              │   │
│  │                  METADATA AGENT                             │   │
│  │  • Retrieves hotel info by hotel_id                        │   │
│  │  • Caches hotel name, description, details                 │   │
│  │  • Enriches query with hotel context                       │   │
│  │  • Tracks conversation history                             │   │
│  └──────────────────────┬─────────────────────────────────────┘   │
│                         │                                           │
│                         ▼                                           │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │                    SUPERVISOR                               │   │
│  │  Intent Detection:                                          │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │   │
│  │  │ Review?  │  │ Booking? │  │ Other?   │                │   │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                │   │
│  └───────┼─────────────┼─────────────┼───────────────────────┘   │
│          │             │             │                             │
│          ▼             ▼             ▼                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                    │
│  │ REVIEW   │  │ BOOKING  │  │ COMPARISON   │                    │
│  │ AGENT    │  │ AGENT    │  │ AGENT        │                    │
│  └──────────┘  └──────────┘  └──────────────┘                    │
│       │             │                │                              │
│       └─────────────┴────────────────┘                             │
│                     │                                               │
│                     ▼                                               │
│                   END                                               │
└───────────────────────────────────────────────────────────────────┘
```

---

## Detailed Agent Flow

### 1. Metadata Agent (Entry Point)

```
┌─────────────────────────────────────────────────────────────┐
│                     METADATA AGENT                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: state["hotel_id"] = "123"                          │
│         state["messages"] = [user query]                    │
│                                                              │
│  ┌────────────────────────────────────────────────┐        │
│  │ 1. Check conversation context cache            │        │
│  │    • Is hotel_id already loaded?               │        │
│  │    • Has hotel info been retrieved?            │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 2. Retrieve hotel information                  │        │
│  │    • Read from hotels.csv by hotel_id          │        │
│  │    • Extract: name, rating, address, desc      │        │
│  │    • Cache in conversation_context             │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 3. Enrich user query                           │        │
│  │    "amenities" → "amenities (Hotel Name)"      │        │
│  │    Contextual keywords trigger enrichment      │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 4. Update state metadata                       │        │
│  │    metadata["hotel_id"] = "123"                │        │
│  │    metadata["hotel_name"] = "Hotel ABC"        │        │
│  │    metadata["hotel_info"] = {...}              │        │
│  │    metadata["resolved_query"] = enriched       │        │
│  └────────────────────────────────────────────────┘        │
│                                                              │
│  Output: state with full hotel context → Supervisor        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 2. Supervisor (Router)

```
┌─────────────────────────────────────────────────────────────┐
│                       SUPERVISOR                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: state["metadata"]["resolved_query"]                │
│                                                              │
│  ┌────────────────────────────────────────────────┐        │
│  │           INTENT DETECTION                     │        │
│  │    (Priority: Review > Booking > Comparison)   │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│       ┌──────────────┼──────────────┐                      │
│       ▼              ▼              ▼                      │
│  ┌────────┐    ┌────────┐    ┌────────────┐              │
│  │Review? │    │Booking?│    │Comparison? │              │
│  │        │    │        │    │            │              │
│  │reviews │    │book    │    │amenities   │              │
│  │ratings │    │reserve │    │similar     │              │
│  │feedback│    │        │    │location    │              │
│  └───┬────┘    └───┬────┘    └─────┬──────┘              │
│      │             │               │                       │
│      ▼             ▼               ▼                       │
│  "review"      "booking"      "comparison"                │
│                                                              │
│  state["intent"] = detected_intent                         │
│  state["route"] = detected_intent                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 3A. Review Agent

```
┌─────────────────────────────────────────────────────────────┐
│                       REVIEW AGENT                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: hotel_id, user query                                │
│                                                              │
│  ┌────────────────────────────────────────────────┐        │
│  │ 1. Retrieve reviews from Pinecone              │        │
│  │    • Query Pinecone Reviews Index              │        │
│  │    • Get all relevant review documents         │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 2. Filter by hotel_id                          │        │
│  │    for doc in reviews:                         │        │
│  │        if doc.metadata["hotel_id"] == hotel_id │        │
│  │            filtered_reviews.append(doc)        │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 3. Detect specific intent                      │        │
│  │    • Recent reviews? → Show 2-3 latest         │        │
│  │    • Summary? → Generate AI summary            │        │
│  │    • General? → Answer based on reviews        │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│            ┌─────────┴─────────┐                           │
│            ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────────┐                │
│  │Recent Reviews│    │  Review Summary  │                │
│  │              │    │                  │                │
│  │**Review 1**  │    │ "Overall, guests │                │
│  │Rating: 4/5   │    │  appreciated the │                │
│  │"Great stay.."│    │  location and... "│               │
│  │              │    │                  │                │
│  │**Review 2**  │    │  LLM-generated   │                │
│  │Rating: 5/5   │    │  from all reviews│                │
│  └──────────────┘    └──────────────────┘                │
│                                                              │
│  Output: Formatted response → END                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 3B. Comparison Agent

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPARISON AGENT                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: hotel_id, user query, hotel_info                    │
│                                                              │
│  ┌────────────────────────────────────────────────┐        │
│  │ 1. Detect query type                           │        │
│  │    • Similar hotels? → Find alternatives       │        │
│  │    • Hotel info? → Answer about current hotel  │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│            ┌─────────┴─────────┐                           │
│            ▼                   ▼                           │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │ Similar Hotels   │  │  Hotel Info      │              │
│  │                  │  │                  │              │
│  │ 1. Get hotel desc│  │ 1. Retrieve docs │              │
│  │ 2. Search Pine-  │  │    from CSV      │              │
│  │    cone for sim- │  │                  │              │
│  │    ilar hotels   │  │ 2. Filter to     │              │
│  │ 3. Filter out    │  │    exact hotel_id│              │
│  │    current hotel │  │    exact hotel_id│              │
│  │ 4. Return top 3  │  │                  │              │
│  │                  │  │ 3. Generate      │              │
│  │ Format:          │  │    answer with   │              │
│  │ "**Hotel A**     │  │    LLM           │              │
│  │  ⭐⭐⭐⭐          │  │                  │              │
│  │  Description...  │  │ "This hotel      │              │
│  │  [Link]          │  │  offers..."      │              │
│  │                  │  │                  │              │
│  │ **Hotel B**      │  │                  │              │
│  │  ..."            │  │                  │              │
│  └──────────────────┘  └──────────────────┘              │
│                                                              │
│  Output: Response → END                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

### 3C. Booking Agent

```
┌─────────────────────────────────────────────────────────────┐
│                      BOOKING AGENT                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: hotel_id, hotel_info, user query                    │
│                                                              │
│  ┌────────────────────────────────────────────────┐        │
│  │ 1. Extract hotel information from state       │        │
│  │    • hotel_id from state                       │        │
│  │    • hotel_name from metadata                  │        │
│  │    • star_rating from hotel_info               │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 2. Create booking record                       │        │
│  │    {                                            │        │
│  │      "thread_id": "...",                       │        │
│  │      "hotel_id": "123",                        │        │
│  │      "hotel_name": "Hotel ABC",                │        │
│  │      "star_rating": "4",                       │        │
│  │      "raw_request": "Book for 2 nights"        │        │
│  │    }                                            │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 3. Persist to bookings_log                     │        │
│  │    • Append to in-memory list                  │        │
│  │    • Write to booking_requests.json            │        │
│  └────────────────────────────────────────────────┘        │
│                      │                                       │
│                      ▼                                       │
│  ┌────────────────────────────────────────────────┐        │
│  │ 4. Generate confirmation                       │        │
│  │    "Great! I've created a booking for          │        │
│  │     {hotel_name}. Next steps: ..."             │        │
│  └────────────────────────────────────────────────┘        │
│                                                              │
│  Output: Confirmation message → END                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Stores

```
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐        │
│  │  VECTOR DATABASES    │  │  STRUCTURED DATA     │        │
│  │                      │  │                      │        │
│  │ Pinecone Hotels Idx  │  │ hotels.csv           │        │
│  │ ├─ hotel_info        │  │ ├─ hotel details     │        │
│  │ └─ metadata          │  │ └─ attributes        │        │
│  │                      │  │                      │        │
│  │ Pinecone Reviews Idx │  └──────────────────────┘        │
│  │ ├─ review_text       │                                   │
│  │ └─ ratings           │  ┌──────────────────────┐        │
│  └──────────────────────┘  │  JSON FILES          │        │
│                             │                      │        │
│  ┌──────────────────────┐  │ booking_requests.json│        │
│  │  IN-MEMORY STORES    │  │ └─ booking records   │        │
│  │                      │  └──────────────────────┘        │
│  │ conversation_context │                                   │
│  │ └─ {thread_id: {...}}│                                   │
│  │                      │                                   │
│  │ last_suggestions     │                                   │
│  │ └─ {thread_id: [...]}│                                   │
│  │                      │                                   │
│  │ bookings_log         │                                   │
│  │ └─ [booking records] │                                   │
│  └──────────────────────┘                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## State Transitions

```
Initial State:
┌──────────────────────────────────────┐
│ messages: [user_query]               │
│ user_id: "user123"                   │
│ thread_id: "thread_abc"              │
│ hotel_id: "123"                      │ ← NEW!
│ intent: None                         │
│ route: None                          │
│ metadata: {}                         │
└──────────────────────────────────────┘
            ↓
After Metadata Agent:
┌──────────────────────────────────────┐
│ messages: [user_query]               │
│ user_id: "user123"                   │
│ thread_id: "thread_abc"              │
│ hotel_id: "123"                      │
│ intent: None                         │
│ route: "supervisor"                  │
│ metadata: {                          │
│   hotel_id: "123",                   │
│   hotel_name: "Hotel ABC",           │
│   hotel_info: {...},                 │
│   resolved_query: "enriched query"   │
│ }                                    │
└──────────────────────────────────────┘
            ↓
After Supervisor:
┌──────────────────────────────────────┐
│ ...                                  │
│ intent: "review"                     │ ← Set by supervisor
│ route: "review"                      │ ← Set by supervisor
│ ...                                  │
└──────────────────────────────────────┘
            ↓
After Review Agent:
┌──────────────────────────────────────┐
│ messages: [                          │
│   user_query,                        │
│   assistant_response                 │ ← Added by agent
│ ]                                    │
│ ...                                  │
│ route: "end"                         │ ← Set by agent
└──────────────────────────────────────┘
```

---

## Conversation Context Management

```
conversation_context[thread_id] = {
  ┌─────────────────────────────────────┐
  │ "hotel_id": "123",                  │ ← Current hotel
  │ "hotel_name": "Hotel ABC",          │ ← Cached name
  │ "hotel_info": {                     │ ← Full hotel data
  │   "name": "Hotel ABC",              │
  │   "description": "...",             │
  │   "star_rating": "4",               │
  │   "address": "..."                  │
  │ },                                  │
  │ "questions": [                      │ ← User queries
  │   "What amenities?",                │
  │   "Show reviews"                    │
  │ ],                                  │
  │ "conversation_pairs": [             │ ← Full conversation
  │   ("What amenities?", "The hotel offers..."),
  │   ("Show reviews", "Here are...")   │
  │ ]                                   │
  └─────────────────────────────────────┘
}
```

---

## Key Architectural Decisions

### ✅ Why Hotel ID First?
- **Eliminates ambiguity**: No confusion about which hotel
- **Better filtering**: Direct hotel_id lookup is faster
- **Clearer context**: Always know what we're talking about
- **Scalability**: Easy to add hotel-specific features

### ✅ Why Separate Review Agent?
- **Specialized functionality**: Reviews need different handling
- **Clear separation**: Not mixed with general info
- **Easy to extend**: Can add review filters, sorting
- **Better organization**: Each agent has clear purpose

### ✅ Why Metadata Agent First?
- **Context loading**: Ensures hotel info available to all agents
- **Query enrichment**: Improves retrieval quality
- **Caching**: Avoids repeated lookups
- **Consistent state**: All agents work with same data

---

## Performance Characteristics

```
Query Flow Timing (Approximate):
────────────────────────────────

API Endpoint         ~5ms
  ↓
Chat Service         ~10ms
  ↓
Metadata Agent       ~100ms  (DB lookup + caching)
  ↓
Supervisor           ~50ms   (Intent detection)
  ↓
Specialized Agent    ~500ms  (LLM + retrieval)
  ↓
Response             ~10ms
─────────────────────────────
Total:               ~675ms

Optimizations:
• Metadata agent caches hotel info (subsequent queries ~50ms faster)
• Vector DB uses HNSW index (fast similarity search)
• Conversation context cached in memory (no DB lookup)
```

---

## Scalability Considerations

### Current Architecture Supports:
- ✅ Multiple concurrent users (thread-based isolation)
- ✅ Multiple hotels (hotel_id separation)
- ✅ Long conversations (message history management)
- ✅ Context persistence (MemorySaver checkpointer)

---

**This architecture provides a solid foundation for a hotel-specific chatbot with clear agent responsibilities and efficient data flow.**

