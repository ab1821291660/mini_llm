"""
MiniAgent Data & Tokenizer Setup
Downloads minimind's tokenizer (6400 vocab) and generates advertising training data.
Usage:
    python scripts/data_advertising.py --all          # tokenizer + all data
    python scripts/data_advertising.py --tokenizer     # just tokenizer
    python scripts/data_advertising.py --base-model    # minimind pretrained weights (for transfer learning)
    python scripts/data_advertising.py --pretrain      # pretrain corpus
    python scripts/data_advertising.py --sft           # SFT instruction pairs
    python scripts/data_advertising.py --dpo           # DPO preference pairs
"""
import os
import json
import argparse
def ________1():pass
"""
MiniAgent Platform Training Data — All 60+ Advertising Platforms

Covers: Search, Social, DSP, Retail/Commerce, Video/CTV, Mobile/App,
Native/Content, Tag/Analytics, Measurement, Affiliate platforms.
"""
PLATFORM_PRETRAIN_TEXTS = [
    # ═══════════════════════════════════════════════════════════
    # SEARCH PLATFORMS
    # ═══════════════════════════════════════════════════════════

    # Google Ads (v23)
    "Google Ads API v23 is the current version, accessed via the google-ads Python package (v25.1+). Authentication requires OAuth 2.0 credentials plus a developer token. The API supports campaign management, reporting via GAQL, keyword planning, and conversion tracking. Rate limit is 15,000 operations per day for basic access.",
    "Google Ads campaign types: Search, Display, Shopping, Video (YouTube), Performance Max, App, Demand Gen, and Local. Each campaign type has unique targeting, bidding, and creative requirements. Performance Max uses AI across all channels simultaneously.",
    "Google Ads Smart Bidding uses machine learning to optimize bids in real-time. Strategies include Target CPA, Target ROAS, Maximize Conversions, and Maximize Conversion Value. Smart Bidding requires at least 30 conversions in 30 days per campaign for effective optimization.",
    "Google Ads Quality Score is rated 1-10 per keyword with three components: Expected CTR, Ad Relevance, and Landing Page Experience. Higher Quality Score reduces CPC. Ad Rank = Max CPC Bid × Quality Score + expected impact of extensions.",
    "GAQL (Google Ads Query Language) uses SELECT-FROM-WHERE-ORDER BY-LIMIT syntax. Key resources: campaign, ad_group, ad_group_ad, keyword_view, search_term_view. Monetary values use cost_micros (divide by 1,000,000). Date filtering uses segments.date DURING LAST_30_DAYS or BETWEEN clauses.",

    # Google Ads Scripts
    "Google Ads Scripts use JavaScript to automate account tasks without API setup. Scripts run on a schedule with a 30-minute execution limit. Common scripts: bid adjustments, broken URL detection, Quality Score tracking, search term mining, budget pacing alerts, and automated reporting. No developer token required.",

    # Google Merchant Center
    "Google Merchant Center hosts product data feeds for Shopping and Performance Max campaigns. The Content API v2.1 enables programmatic feed management. Required fields: id, title, description, link, image_link, price, availability, brand, gtin or mpn, condition. Feed quality directly impacts Shopping ad performance.",
    "Merchant Center feed optimization: titles should include brand + product type + key attribute + color/size. Use Supplemental feeds for quick updates without modifying the primary feed. Custom labels (custom_label_0 through custom_label_4) enable campaign segmentation by margin, seasonality, or priority.",

    # Google Ad Manager
    "Google Ad Manager (GAM) is an ad serving platform for publishers managing direct-sold and programmatic inventory. It supports display, video, native, and audio ad formats. The API uses OAuth 2.0 for authentication. GAM integrates with AdSense, Ad Exchange, and third-party demand sources.",

    # Google Campaign Manager 360
    "Campaign Manager 360 (CM360) is Google's ad server and measurement platform for agencies and advertisers. It provides cross-channel campaign management, trafficking, verification, and reporting. Uses Floodlight tags for conversion tracking. The API enables automated campaign trafficking and reporting.",

    # Google Display & Video 360
    "Display & Video 360 (DV360) is Google's demand-side platform for programmatic media buying across display, video, audio, native, and connected TV. It integrates with Campaign Manager 360 for ad serving. Uses OAuth 2.0 authentication. Supports automated bidding, audience management, and creative optimization.",

    # Microsoft Advertising (Bing)
    "Microsoft Advertising (formerly Bing Ads) uses the bingads Python SDK with OAuth 2.0 authentication. It reaches 36% of US desktop search with 30-50% lower CPCs than Google. Key differentiator: LinkedIn profile targeting for B2B (job title, company, industry). Supports direct campaign import from Google Ads.",
    "Microsoft Advertising UET (Universal Event Tracking) tag tracks conversions and remarketing. The API supports campaign management, reporting, and bulk operations. Smart Bidding strategies mirror Google: Target CPA, Target ROAS, Enhanced CPC, Maximize Conversions.",

    # Apple Search Ads
    "Apple Search Ads places ads in App Store search results. Two tiers: Basic (automated, pay-per-install) and Advanced (keyword bidding, audience targeting). The API uses OAuth 2.0. Key metrics: tap-through rate (TTR), conversion rate (CR), cost per acquisition (CPA). Apple Search Ads Attribution API measures install attribution.",

    # Yahoo/Verizon DSP
    "Yahoo DSP (formerly Verizon Media DSP) provides programmatic buying across Yahoo properties and the open web. Supports display, native, video, and connected TV. Uses Deal ID-based buying and open auction. Access requires a managed account relationship.",

    # ═══════════════════════════════════════════════════════════
    # SOCIAL PLATFORMS
    # ═══════════════════════════════════════════════════════════

    # Meta (Facebook/Instagram)
    "Meta Ads uses the facebook-business Python SDK (v21.0+) with System User Token or User Token authentication. Campaign structure: Campaign > Ad Set > Ad. Campaign Budget Optimization (CBO) distributes budget across ad sets. Advantage+ campaigns automate targeting and placement.",
    "Meta Conversions API (CAPI) sends server-side events directly to Meta, complementing the Pixel for iOS 14.5+ tracking recovery. Standard events: PageView, ViewContent, AddToCart, InitiateCheckout, Purchase, Lead, CompleteRegistration. Requires System User Token authentication.",
    "Meta Ads Library API provides transparency data on ads running across Facebook and Instagram. Access requires an app with ads_read permission. Returns ad creative, spend ranges, impressions, and demographic targeting. Useful for competitive intelligence.",
    "Meta Pixel fires browser-side events for conversion tracking and audience building. Aggregated Event Measurement limits optimization to 8 events per domain. Custom Audiences from website visitors, customer lists, app activity, and video views. Lookalike Audiences range from 1% (most similar) to 10%.",

    # LinkedIn Marketing API
    "LinkedIn Marketing API uses OAuth 2.0 with granular permission scopes. Campaign structure: Campaign Group > Campaign > Creative. Unique targeting: job title, company name, industry, seniority, skills, company size, and LinkedIn Groups. Lead Gen Forms pre-populate with profile data for 2-5x higher conversion rates.",
    "LinkedIn Ads CPCs typically range $5-15+ due to premium B2B audience. Campaign objectives: Brand Awareness, Website Visits, Engagement, Video Views, Lead Generation, Website Conversions. Sponsored Content, Message Ads, Dynamic Ads, and Text Ads are the main formats.",

    # X/Twitter Ads API
    "X Ads API (formerly Twitter Ads) uses OAuth 1.0a + OAuth 2.0 dual authentication. Campaign structure mirrors standard hierarchy. Targeting includes keywords, interests, followers of specific accounts, tailored audiences (customer lists), and conversation topics. Promoted Tweets and Promoted Trends are key formats.",

    # TikTok Business API
    "TikTok Business API uses App ID + Secret authentication. Campaign structure: Campaign > Ad Group > Ad. The Business API rate limit is 10 requests per second. Spark Ads boost organic content as paid ads. Smart Performance Campaigns are TikTok's fully automated campaign type.",
    "TikTok Ads targeting includes interest categories, behavior targeting, custom audiences (website, app, customer file), and lookalike audiences. TikTok Shop integrates ecommerce directly. Creative best practices: vertical video, first 3 seconds hook, authentic creator-style content.",

    # Snapchat Marketing API
    "Snapchat Marketing API uses OAuth 2.0 authentication. Ad formats: Single Image/Video, Collection Ads, Story Ads, AR Lenses, Filters, and Commercials. Targeting includes demographics, interests, Snap Lifestyle Categories, Custom Audiences, and Lookalike Audiences. The Snap Pixel tracks website conversions.",

    # Pinterest Ads API
    "Pinterest Ads API uses OAuth 2.0. Campaign structure: Campaign > Ad Group > Ad (Pin). Ad formats: Standard Pins, Video Pins, Shopping Pins, Carousel Pins, Collections, and Idea Pins. Targeting includes interests, keywords, demographics, customer lists, actalikes (Pinterest's lookalikes), and retargeting.",
    "Pinterest conversion tracking uses the Pinterest Tag (JavaScript) and Conversions API. Shopping campaigns connect product catalogs. Pinterest users have high purchase intent — 85% of weekly Pinners have purchased based on Pins they've seen.",

    # Reddit Ads API
    "Reddit Ads API uses OAuth 2.0 with API Key authentication. Targeting includes subreddit/community targeting, interest targeting, and conversation placement. Promoted posts appear in feeds. CPCs typically $0.50-3. Success requires authentic language matching subreddit culture. The python-reddit package provides API access.",

    # Quora Ads API
    "Quora Ads API provides access to Quora's question-and-answer platform advertising. Targeting includes topic targeting, question targeting, audience targeting, and keyword targeting. Ad formats: Text Ads, Image Ads, and Promoted Answers. Conversion tracking uses the Quora Pixel.",

    # Threads API
    "Threads API (by Meta) is in early development. The platform supports text-based posts up to 500 characters with images and video. Advertising capabilities are being developed. API access follows Meta's developer platform authentication patterns.",

    # ═══════════════════════════════════════════════════════════
    # DSP / PROGRAMMATIC
    # ═══════════════════════════════════════════════════════════

    # The Trade Desk
    "The Trade Desk (TTD) is the largest independent DSP for programmatic advertising. API uses API Key authentication. Supports display, video, audio, native, CTV, and DOOH inventory. Uses Koa AI for bid optimization. Unified ID 2.0 is their cookieless identity solution. Requires $50K+ monthly minimum spend.",

    # Amazon DSP
    "Amazon DSP enables programmatic buying on and off Amazon properties. Uses AWS-based authentication (Login with Amazon). Reaches Amazon shoppers with exclusive first-party purchase data. Supports display, video, and audio. Self-serve minimum $35K+. Available audiences: in-market, lifestyle, and Amazon purchase-based segments.",

    # DV360 (Google)
    "DV360 (Display & Video 360) is Google's enterprise DSP with access to Google Ad Exchange inventory plus 80+ third-party exchanges. Uses OAuth 2.0. Supports Programmatic Guaranteed, Preferred Deals, Private Auctions, and Open Auction buying methods. Integrates with CM360 and Google Analytics.",

    # Xandr (Microsoft)
    "Xandr (now Microsoft Advertising DSP) provides programmatic buying across display, video, native, and CTV. Uses username/password token authentication. Offers both self-serve and managed service tiers. Integrates with Microsoft's audience data including LinkedIn signals.",

    # MediaMath
    "MediaMath is an enterprise DSP specializing in omnichannel programmatic advertising. Uses API Key authentication. Supports display, video, native, audio, CTV, and DOOH. Features include audience management, identity resolution, and cross-device targeting.",

    # Criteo
    "Criteo is a commerce media platform specializing in retargeting and retail media. Uses OAuth 2.0 with the criteo-api-python-sdk. Products include dynamic retargeting (automatically shows products users viewed), retail media (ads on retailer sites), and commerce audiences. Known for high-quality product recommendation algorithms.",

    # AdRoll / RollWorks
    "AdRoll provides retargeting, prospecting, and email marketing for D2C and ecommerce. RollWorks is their B2B account-based marketing platform. Uses OAuth 2.0 with API Key. Features cross-channel retargeting across web, social, and email. Integrates with Shopify, WooCommerce, and major ecommerce platforms.",

    # StackAdapt
    "StackAdapt is a self-serve programmatic DSP for native, display, video, CTV, audio, and in-game advertising. Uses API Key authentication. Known for strong native advertising capabilities. Features contextual targeting, audience targeting, and page-context AI for brand safety.",

    # Basis Technologies (Centro)
    "Basis Technologies (formerly Centro) provides a programmatic advertising platform combining DSP, workflow automation, and business intelligence. Supports display, video, native, audio, CTV, and DOOH. The platform unifies programmatic, direct, search, and social campaign management.",

    # ═══════════════════════════════════════════════════════════
    # RETAIL / COMMERCE
    # ═══════════════════════════════════════════════════════════

    # Amazon Ads (Sponsored)
    "Amazon Sponsored Products, Sponsored Brands, and Sponsored Display are the three core ad formats. ACOS (Advertising Cost of Sales) = Ad Spend / Ad Revenue is the primary metric. Automatic targeting discovers relevant search terms. Manual targeting uses keyword or product targeting. Uses amazon-ads Python SDK v5.0+.",
    "Amazon Attribution measures how non-Amazon channels (Google, Meta, email) drive Amazon sales. Uses Login with Amazon authentication. Provides click-through and conversion data with 14-day attribution window. Essential for brands running off-Amazon media driving to Amazon product pages.",

    # Walmart Connect
    "Walmart Connect is Walmart's retail media platform. Offers Sponsored Products (search), Display (on-site and off-site), and In-Store advertising. Uses Client ID + Secret authentication. Requires REST API integration. Leverages Walmart's first-party purchase data for targeting and closed-loop measurement.",

    # Instacart Ads
    "Instacart Ads offers Sponsored Products and Display advertising on the Instacart grocery delivery platform. Features include keyword targeting, automated bidding, and brand pages. Uses OAuth 2.0 authentication. Provides closed-loop measurement connecting ad impressions to purchases.",

    # Shopify Audiences
    "Shopify Audiences uses aggregated commerce data from Shopify merchants to create high-intent audience lists for advertising on Meta, Google, Pinterest, Snapchat, TikTok, and Criteo. Available to Shopify Plus merchants. Audiences update automatically based on purchase signals.",

    # Google Shopping (CSS)
    "Google Comparison Shopping Services (CSS) allow third-party shopping comparison sites to place Shopping ads on behalf of merchants. CSS partners can offer reduced auction fees compared to Google Shopping directly. Multiple CSS partners can represent the same merchant in different markets.",

    # eBay Promoted Listings
    "eBay Promoted Listings offers Standard (cost-per-sale) and Advanced (cost-per-click) ad formats. Standard Listings charge only when an item sells after an ad click. The API enables automated listing promotion and performance reporting. No minimum spend required.",

    # Target Roundel
    "Target Roundel is Target's retail media network offering display, video, social, and search advertising. Uses Target's first-party Guest ID data covering 165M+ shoppers. Provides closed-loop measurement connecting ad exposure to in-store and online purchases. Access is through a managed service or partner agreement.",

    # ═══════════════════════════════════════════════════════════
    # VIDEO / CTV
    # ═══════════════════════════════════════════════════════════

    # YouTube Ads (via Google Ads)
    "YouTube advertising runs through Google Ads using Video campaigns. Formats: skippable in-stream, non-skippable in-stream, bumper (6 sec), in-feed video, and Shorts ads. Targeting includes demographics, interests, topics, keywords, placements, and custom audiences. Uses google-ads Python library with google-py-video-youtube.",

    # Roku OneView
    "Roku OneView (formerly dataxu) is Roku's advertising platform for CTV and streaming. Provides access to Roku's first-party viewership data across 80M+ active accounts. Supports video, display, and native formats on Roku devices and the broader CTV ecosystem. Access through partner agreement.",

    # Samsung Ads
    "Samsung Ads leverages Samsung Smart TV data for CTV advertising and audience targeting. Offers Native Samsung TV+ ads, display ads, and programmatic buying via partner DSPs. Uses Samsung's Automatic Content Recognition (ACR) data for viewership insights and targeting.",

    # Spotify Ad Studio
    "Spotify Ad Studio is the self-serve platform for audio and video ads on Spotify. Targets by demographics, interests, listening behavior, playlists, and podcasts. Audio ads can be generated using Spotify's text-to-speech tool. Minimum campaign spend starts at $250.",

    # ═══════════════════════════════════════════════════════════
    # MOBILE / APP
    # ═══════════════════════════════════════════════════════════

    # Google AdMob
    "Google AdMob monetizes mobile apps with display, interstitial, rewarded, and native ads. The AdMob API uses OAuth 2.0. Integrates with Google Ads for app campaign management. Supports mediation to optimize fill rates across multiple ad networks. Firebase integration enables analytics and A/B testing.",

    # Unity Ads
    "Unity Ads provides in-game advertising for Unity-developed games. Supports rewarded video, interstitial, and banner formats. The SDK integrates into Unity game engine projects. Uses Dashboard Key authentication. Particularly strong in the gaming vertical with rewarded video monetization.",

    # AppsFlyer
    "AppsFlyer is a mobile attribution and analytics platform. Tracks app installs and in-app events across advertising channels. Uses API Key authentication. Features include deep linking, fraud protection, incrementality measurement, and audience segmentation. Integrates with 10,000+ technology partners.",

    # Adjust
    "Adjust provides mobile attribution, analytics, and fraud prevention. Tracks installs, events, and revenue across channels. Uses API Key authentication with the adjust-api Python SDK. Features include deep linking, audience builder, incrementality testing, and automated fraud prevention.",

    # AppLovin
    "AppLovin MAX is a mediation platform that optimizes ad revenue across networks. AppLovin's AppDiscovery uses machine learning for user acquisition. The ALX exchange enables programmatic in-app advertising. Uses Dashboard Key authentication.",

    # Branch (Attribution)
    "Branch provides deep linking and mobile attribution. Creates universal links that route users to the right content across web, app, and email. Measures cross-platform attribution. Uses API Key authentication. Integrates with all major ad networks and analytics platforms.",

    # ═══════════════════════════════════════════════════════════
    # NATIVE / CONTENT
    # ═══════════════════════════════════════════════════════════

    # Taboola
    "Taboola is a native advertising platform reaching 500M+ daily users across premium publisher sites. Uses OAuth 2.0 authentication. Ad formats include native content recommendations, video, and display. Targeting includes audience segments, contextual, and custom audiences. Known for content discovery and recommendation widgets.",

    # Outbrain
    "Outbrain powers native advertising on premium publisher websites. Uses API Key authentication. Supports native content, video, carousel, and app install formats. Targeting includes interest-based, contextual, lookalike, and custom audience segments. Merged with Taboola in recent years.",

    # MGID
    "MGID is a native advertising platform providing content recommendations across global publishers. Supports native, display, and video formats. Uses API Key authentication. Features include contextual targeting, audience segments, and AI-powered optimization. Known for strong presence in international markets.",

    # ═══════════════════════════════════════════════════════════
    # TAG / ANALYTICS
    # ═══════════════════════════════════════════════════════════

    # Google Analytics 4 (GA4)
    "Google Analytics 4 (GA4) uses event-based tracking replacing Universal Analytics. The GA4 API uses OAuth 2.0 with the google-analytics-data Python package. Key features: cross-platform tracking, enhanced measurement, BigQuery export, predictive audiences, and privacy-centric design with cookieless measurement.",
    "GA4 event model: automatically collected (page_view, first_visit), enhanced measurement (scroll, outbound_clicks, file_downloads), recommended events (purchase, add_to_cart, sign_up), and custom events. Up to 500 distinct events. Conversion events are marked in the UI.",

    # Google Tag Manager
    "Google Tag Manager (GTM) manages marketing and analytics tags without code changes. Uses OAuth 2.0 with the tagmanager v2 API. Supports web and mobile containers. Server-side GTM runs tags on a cloud server for improved performance and data control. Triggers fire tags based on events, page views, or custom conditions.",

    # Google Search Console
    "Google Search Console provides organic search performance data: impressions, clicks, CTR, and average position by query, page, country, and device. The API uses OAuth 2.0 with google-api-python-client. Essential for SEO and understanding how paid and organic search interact.",

    # Meta Pixel / CAPI
    "Meta Pixel (client-side) and Conversions API (server-side) work together for complete conversion tracking. Post-iOS 14.5, server-side CAPI is essential for accurate measurement. Aggregated Event Measurement limits optimization to 8 priority events per domain. Event Match Quality score measures data signal strength.",

    # Microsoft Clarity
    "Microsoft Clarity is a free user behavior analytics tool providing session recordings, heatmaps, and insights. No sampling and no data limits. Integrates with Google Analytics and Microsoft Advertising. Useful for diagnosing landing page issues affecting ad conversion rates.",

    # ═══════════════════════════════════════════════════════════
    # MEASUREMENT
    # ═══════════════════════════════════════════════════════════

    # Google Ads Data Hub
    "Google Ads Data Hub enables privacy-safe analysis of Google Ads campaign data in BigQuery. Combines Google's ad event data with advertiser's first-party data. Uses SQL for custom analysis. No private API — accessed through Google Cloud. Essential for enterprise-level campaign attribution and audience analysis.",

    # Google Ads Transparency Center
    "Google Ads Transparency Center shows verified advertiser information and ads served across Google platforms. No public API. Useful for competitive research — see what ads competitors are running, their verification status, and regions where ads are served.",

    # ═══════════════════════════════════════════════════════════
    # AFFILIATE / OTHER
    # ═══════════════════════════════════════════════════════════

    # Commission Junction (CJ)
    "Commission Junction (CJ Affiliate) is one of the largest affiliate marketing networks. Uses API Key authentication. Provides publisher and advertiser APIs for program management, reporting, and link generation. Tracks conversions through CJ's tracking pixel and server-side postback.",

    # ShareASale
    "ShareASale (now part of Awin) is an affiliate marketing network connecting merchants with publishers. Uses API Key authentication via ShareASale API Reference. Provides reporting, merchant discovery, and affiliate link management APIs.",

    # Impact
    "Impact is a partnership automation platform for affiliate, influencer, and B2B partnerships. Uses API Key authentication. Features include partner discovery, contracting, tracking, payment, and optimization. Supports mobile app attribution and cross-device tracking.",

    # Rakuten Advertising
    "Rakuten Advertising provides affiliate marketing, display, and search retargeting services. Uses API Token authentication. The network connects brands with publishers globally. Features include commissioning, reporting, and creative management APIs.",

    # ═══════════════════════════════════════════════════════════
    # CROSS-PLATFORM KNOWLEDGE
    # ═══════════════════════════════════════════════════════════

    "OAuth 2.0 is the most common authentication method across advertising platforms. The flow: register an app, get client credentials, redirect user for authorization, receive an access token, refresh when expired. Google, Meta, Microsoft, LinkedIn, Snapchat, Pinterest, and Criteo all use OAuth 2.0.",
    "API Key authentication is simpler than OAuth — a single key in the request header. Used by The Trade Desk, Reddit, MediaMath, Taboola, Outbrain, CJ Affiliate, and ShareASale. Less secure than OAuth but easier to implement for server-to-server integrations.",
    "Retail Media Networks (RMNs) are the fastest-growing advertising channel. Amazon, Walmart, Target, Instacart, and Kroger all offer advertising on their commerce platforms. Key advantage: closed-loop measurement connecting ad exposure directly to purchase. RMN spend exceeded $40B in 2024.",
    "Connected TV (CTV) advertising reaches viewers on streaming platforms. Available through DV360, The Trade Desk, Amazon DSP, Roku OneView, Samsung Ads, and platform-specific buys. CTV offers TV-like reach with digital targeting precision. Key metrics: completed views, reach, frequency, and brand lift.",
    "Mobile Measurement Partners (MMPs) like AppsFlyer, Adjust, Branch, and Singular provide app install attribution. They determine which ad click or view led to an app install using device identifiers and probabilistic modeling. Essential for mobile app advertising across all platforms.",
    "Server-side tracking (Meta CAPI, Google Enhanced Conversions, LinkedIn CAPI, TikTok Events API) sends conversion data directly from your server to the ad platform. This bypasses browser restrictions (ITP, cookie blocking) and improves attribution accuracy. Increasingly required post-iOS 14.5.",
    "Cross-platform metric normalization: Google's CPA = Meta's Cost Per Result = LinkedIn's Cost Per Conversion. Google's ROAS = Amazon's inverse ACOS (1/ACOS × 100). Google's CTR = most platforms' CTR. Create a unified reporting layer mapping each platform's terminology to standard KPIs.",
    "Programmatic advertising uses automated technology to buy and sell digital ad inventory in real-time. The ecosystem includes DSPs (buy-side), SSPs (sell-side), ad exchanges (marketplaces), and DMPs (data management). RTB (real-time bidding) auctions happen in under 100 milliseconds.",
    "First-party data activation: upload customer lists to Google (Customer Match), Meta (Custom Audiences), LinkedIn (Matched Audiences), TikTok (Custom Audiences), and Pinterest (Customer Lists). Match rates vary: email 30-60%, phone 20-40%. Hash data (SHA-256) before upload for privacy compliance.",
    "Privacy regulations impact advertising: GDPR (EU), CCPA/CPRA (California), LGPD (Brazil). All platforms require consent for tracking. Cookie deprecation affects cross-site tracking. Solutions: server-side tracking, first-party data strategies, contextual targeting, and cohort-based solutions.",
    "Multi-touch attribution models: Last Click (default for most platforms, gives all credit to final touchpoint), Data-Driven (ML-based, Google and Meta default), Time Decay (more credit to recent interactions), Position-Based (40/20/40 split first/middle/last). Each platform tends to over-credit its own channel.",
    "Creative asset specifications vary by platform. Google Responsive Display: images 1200×628, 1200×1200; headlines 30 chars; descriptions 90 chars. Meta: images 1080×1080 or 1200×628; primary text 125 chars; headline 40 chars. LinkedIn: images 1200×627; headline 70 chars. TikTok: 9:16 vertical video, 5-60 seconds.",
]


PLATFORM_SFT_PAIRS = [
    # Cross-platform scenarios
    {"user": "What advertising platforms should a B2B SaaS company use?",
     "assistant": "For B2B SaaS, prioritize these platforms: 1) Google Ads Search (50-60% budget) — capture high-intent users searching for solutions. Use Target CPA bidding with offline conversion tracking. 2) LinkedIn Ads (20-25%) — unmatched B2B targeting by job title, company, industry. Lead Gen Forms get 2-5x higher conversion rates. CPCs are $5-15+ but lead quality is premium. 3) Microsoft Ads (10-15%) — import from Google, get 30-50% lower CPCs with older/higher-income audience + LinkedIn targeting integration. 4) Meta Ads (5-10%) — retargeting warm audiences and lookalikes from customer lists. Start with Google + LinkedIn, prove ROI, then expand."},

    {"user": "How do I track conversions across Google, Meta, and LinkedIn?",
     "assistant": "Set up three tracking systems: 1) Google — install gtag.js or use Google Tag Manager, configure conversion actions in Google Ads, enable Enhanced Conversions (sends hashed email server-side). 2) Meta — install Meta Pixel for client-side events, set up Conversions API (CAPI) for server-side events, configure Aggregated Event Measurement with your 8 priority events. 3) LinkedIn — install LinkedIn Insight Tag, set up conversion tracking with URL or event-based rules. For all three: use UTM parameters consistently, set up Google Analytics 4 as your source of truth to deduplicate cross-platform conversions, and implement server-side tracking for accuracy post-iOS 14.5."},

    {"user": "Explain the difference between a DSP and a social platform for advertising.",
     "assistant": "DSPs (The Trade Desk, DV360, Amazon DSP, Xandr) buy ad inventory programmatically across thousands of websites, apps, and CTV through real-time bidding. They excel at: reach across the open web, advanced audience targeting using third-party data, cross-channel buying (display, video, CTV, audio, DOOH). Social platforms (Meta, LinkedIn, TikTok, Snapchat, Pinterest) sell advertising only on their own properties. They excel at: rich first-party user data, native ad formats that blend with content, social engagement and sharing. Use DSPs for broad reach and retargeting at scale. Use social for awareness, engagement, and leveraging platform-specific audiences and formats."},

    {"user": "What is ACOS on Amazon and how does it compare to Google ROAS?",
     "assistant": "ACOS (Advertising Cost of Sales) = Ad Spend / Ad Revenue × 100. It measures what percentage of revenue went to ads. ROAS (Return on Ad Spend) = Revenue / Ad Spend. They're inverse: ROAS = 1 / (ACOS/100). Examples: ACOS 25% = ROAS 400%. ACOS 50% = ROAS 200%. ACOS 10% = ROAS 1000%. For Amazon, target ACOS below your profit margin. If margin is 30%, keep ACOS under 25% for profitability. On Google, the equivalent is Target ROAS > break-even ROAS (1/margin). Same math, flipped metric."},

    {"user": "How do retail media networks like Walmart Connect and Target Roundel compare to Amazon Ads?",
     "assistant": "All three are retail media networks (RMNs) using first-party purchase data for targeting and closed-loop measurement. Amazon Ads: largest RMN, self-serve via amazon-ads SDK, Sponsored Products/Brands/Display, ACOS-based optimization, 300M+ active customers. Walmart Connect: second-largest, uses Client ID + Secret auth, Sponsored Products + Display, leverages 150M+ weekly shoppers, both on-site and off-site inventory. Target Roundel: managed service primarily, uses Target's Guest ID covering 165M+ shoppers, display, video, social, and search formats, strong in-store measurement. Start with Amazon if you sell there, then expand to Walmart and Target based on your distribution."},

    {"user": "What mobile attribution platforms should I use, and how do they work?",
     "assistant": "The major MMPs: AppsFlyer (largest, broadest integrations), Adjust (strong fraud prevention), Branch (best deep linking), Singular (includes creative analytics). How they work: 1) User clicks an ad on Facebook/Google/TikTok/etc. 2) MMP records the click with device fingerprint or identifier. 3) User installs and opens the app. 4) MMP SDK fires, matches the install to the original click. 5) Attribution is reported to the ad platform. They use IDFA (iOS, requires ATT consent), GAID (Android), and probabilistic modeling when identifiers aren't available. Pick AppsFlyer or Adjust for broad needs. Branch if deep linking is critical. All use API Key authentication."},

    {"user": "How do I advertise on CTV (Connected TV)?",
     "assistant": "CTV advertising reaches streaming viewers on smart TVs, Roku, Fire TV, Apple TV, etc. Your options: 1) DV360 — Google's DSP, access to YouTube TV + programmatic CTV inventory, OAuth 2.0, best for Google ecosystem users. 2) The Trade Desk — largest independent DSP, broad CTV inventory across all platforms, API Key auth, $50K+ minimum. 3) Amazon DSP — Fire TV + Amazon Freevee inventory, AWS auth, strong purchase-based targeting. 4) Roku OneView — direct access to Roku's 80M+ accounts, first-party viewership data. 5) Samsung Ads — Samsung Smart TV ACR data. For most advertisers, start with DV360 or The Trade Desk. CTV campaigns optimize for completed views, reach, and frequency. Expect CPMs of $25-40."},

    {"user": "How do I use Taboola or Outbrain for native advertising?",
     "assistant": "Taboola and Outbrain place native content recommendations on premium publisher sites (CNN, Forbes, Business Insider, etc.). Setup: 1) Create an advertiser account, set up conversion tracking pixel. 2) Upload content — articles, landing pages, or product pages with compelling headlines and thumbnails. 3) Set targeting: by audience interest, contextual relevance, device, geo, or custom audiences. 4) Set CPC bids (typically $0.30-1.50 for B2B, $0.10-0.50 for content). 5) Optimize based on CTR and conversion rate. Key difference from search/social: users discover your content while reading articles, so the content must be genuinely valuable, not just a sales pitch. Taboola uses OAuth 2.0, Outbrain uses API Key authentication."},

    {"user": "What tracking tags should every website have installed?",
     "assistant": "Essential tags: 1) Google Tag Manager — container that manages all other tags. 2) GA4 — event-based analytics, the source of truth. 3) Google Ads gtag — conversion tracking + remarketing. 4) Meta Pixel — conversion tracking + audience building. 5) Meta Conversions API — server-side complement to Pixel. 6) LinkedIn Insight Tag — conversion tracking + audience building (if running LinkedIn Ads). 7) Microsoft UET Tag — if running Microsoft Ads. 8) TikTok Pixel — if running TikTok Ads. Recommended: Microsoft Clarity (free heatmaps/recordings), Google Search Console verification. Install everything through GTM for easy management. Implement server-side tracking (GTM Server Container) for maximum accuracy."},

    {"user": "Compare Google Performance Max, Meta Advantage+, and TikTok Smart Performance campaigns.",
     "assistant": "All three are AI-automated campaign types that handle targeting, placement, and bidding: Google PMax: optimizes across Search, Shopping, Display, YouTube, Discover, Gmail, Maps. Requires asset groups with text + images + video + audience signals. Best with 50+ monthly conversions and a product feed. Meta Advantage+: automates audience targeting and creative combinations. Advantage+ Shopping is catalog-based. Works best with established Pixel data and broad product catalogs. TikTok Smart Performance: fully automated campaigns with minimal setup. Provides creative assets and TikTok handles targeting, bidding, and placement. All three reduce manual control in exchange for algorithmic optimization. They work best with sufficient conversion data (30-50+/month) and strong creative assets. Keep dedicated search/brand campaigns running alongside."},

    {"user": "How do I set up server-side conversion tracking across platforms?",
     "assistant": "Server-side tracking sends conversion data from your server directly to ad platforms, bypassing browser limitations. Setup: 1) Google Tag Manager Server Container — deploy on Google Cloud, configure Google Ads and GA4 clients, route events server-side. 2) Meta Conversions API — send events via HTTP POST to graph.facebook.com, include user parameters (hashed email, phone, IP, user agent) for matching. System User Token authentication. 3) LinkedIn Conversions API — send events with hashed email/company for matching. 4) TikTok Events API — send events to business-api.tiktok.com with hashed user data. For all platforms: hash PII with SHA-256 before sending, include event_time and event_source_url, send both client-side and server-side events with event_id for deduplication."},

    # === Mined from user's repos — practitioner expertise ===

    {"user": "How do I set up primary vs secondary conversions correctly in Google Ads?",
     "assistant": "This is the most misunderstood concept in Google Ads. Primary conversions are used by Smart Bidding for optimization. Secondary conversions are tracked for reporting only. The common mistake: setting everything as primary (page views, button clicks, form starts, completions, purchases) — the algorithm can't optimize for contradictory goals. The fix: set ONE primary conversion per campaign objective (usually Purchase for ecommerce, Lead Submit for lead gen). Set everything else as secondary: Add to Cart, Form Start, Page View, Phone Click. Also configure: attribution model = data-driven, count = One for leads / Every for purchases, click-through window = 30 days (shorten for impulse, lengthen for B2B)."},

    {"user": "How do I set up enhanced conversions in Google Ads?",
     "assistant": "Enhanced conversions improve measurement by sending hashed first-party data. Key gotchas: 1) Enhanced conversions for LEADS vs for WEB are two different features — most advertisers need both. 2) You must accept Customer Data Terms in Google Ads UI (Settings > Measurement) before the API works. 3) SHA-256 hash data BEFORE sending — Google won't re-hash correctly. 4) Email normalization matters: john.smith@gmail.com and John.Smith@Gmail.com produce different hashes. Always lowercase, trim whitespace, remove Gmail dots. Via GTM: use the built-in 'Google Ads User-Provided Data' variable — don't build custom JS. Via API: use ConversionUploadService with pre-hashed user identifiers (email, phone, name, address). Verification takes 48-72 hours in the Diagnostics tab."},

    {"user": "Walk me through a full Google Ads 7-dimension audit.",
     "assistant": "The 7-dimension audit framework: 1) Campaign Structure (15% weight): check naming conventions, campaign type appropriateness, ad groups under 20 keywords, match type progression. 2) Keyword Strategy (20%): relevance to landing pages, match type distribution (too much broad = red flag), negative keyword coverage, Quality Score distribution. 3) Ad Copy (10%): RSA pin usage (over-pinning reduces optimization), ad strength Good or better, extension coverage. 4) Bidding (15%): strategy matches conversion volume, targets align with business goals, learning period respected. 5) Budget (15%): budget lost impression share by campaign, top campaign not underfunded, daily vs monthly pacing. 6) Conversion Tracking (15%): primary vs secondary set correctly, data freshness, attribution model, tags verified. 7) Audiences (10%): geo precision, device adjustments, remarketing lists above 1000. Score each 1-100, create 30/60/90-day action plans."},

    {"user": "How do metrics map across Google, Meta, Microsoft, and Amazon?",
     "assistant": "Key metric mappings: Cost = Google cost_micros (divide by 1M) / Meta spend (dollars) / Microsoft Spend (dollars) / Amazon cost (dollars). Clicks = Google clicks / Meta link_clicks / Microsoft Clicks / Amazon clicks. Conversions = Google conversions / Meta actions / Microsoft Conversions / Amazon purchases. Revenue = Google conversions_value / Meta action_values / Microsoft Revenue / Amazon sales. Bidding strategies: Google TARGET_CPA = Meta COST_CAP = Microsoft TargetCpa. Google TARGET_ROAS = Meta MINIMUM_ROAS = Microsoft TargetRoas. Google MAXIMIZE_CLICKS = Meta LOWEST_COST_WITHOUT_CAP. Account hierarchy: Google Campaign > Ad Group > Ad = Meta Campaign > Ad Set > Ad = Amazon Campaign > Ad Group > Ad. Note: Meta and Amazon don't use keyword match types."},

    {"user": "What are the target success metrics for a well-managed PPC account?",
     "assistant": "Enterprise PPC benchmarks: ROAS/CPA within 2 standard deviations of target. Brand impression share 90%+ (protect your brand). Non-brand top keyword impression share 40-60% (budget permitting). Quality Score: 70%+ of spend on QS 7+ keywords. Budget utilization 95-100% daily pacing with under 5% waste. Conversion volume growing 15-25% quarter-over-quarter at stable efficiency. Account health: under 5% of spend on low-performing or redundant elements. Testing velocity: 2-4 structured tests running per month. New campaigns reaching steady-state performance within 2-3 weeks. Weekly search term reviews, monthly full performance reviews, quarterly strategy reviews."},

    {"user": "How do I classify and approach different types of PPC analysis questions?",
     "assistant": "Four question types with different approaches: 1) Performance check ('how are my campaigns doing?') — pull top-level metrics (spend, conversions, CPA, ROAS, impression share), compare to benchmarks and prior period. 2) Diagnostic ('why is CPA increasing?') — segment by dimension: time (when did it start), device (mobile vs desktop), geo (which locations), keyword (which terms are expensive), audience (which segments underperform). Find the break point. 3) Opportunity ('where can I improve?') — analyze impression share gaps (budget vs rank lost IS), wasted spend in search terms, Quality Score distribution, underperforming ad copy. 4) Anomaly ('something looks wrong') — compare current vs prior period for every metric, check change history, look for disapproved ads, bid strategy changes, or new competitors in Auction Insights."},

    {"user": "What Python packages do I need to manage ads across all major platforms?",
     "assistant": "Core Python packages by platform: google-ads (v25.1+) for Google Ads API v23. facebook-business (v21.0+) for Meta Ads. bingads for Microsoft Advertising. amazon-ads (v5.0+) for Amazon Sponsored Ads. google-analytics-data for GA4 reporting. google-api-python-client for Search Console, GTM, DV360. linkedin-api for LinkedIn (community package). criteo-api-python-sdk for Criteo. For attribution: appsflyer and adjust SDKs. For automation: use requests library for platforms without official SDKs (TikTok, Snapchat, Pinterest, Reddit, Taboola). All are pip-installable. Most use OAuth 2.0 — store credentials securely using environment variables, never hardcode tokens."},

    {"user": "How do I build a unified cross-platform advertising report?",
     "assistant": "Build a unified reporting pipeline: 1) Data extraction — pull data from each platform's API. Use scheduled Python scripts or tools like Supermetrics, Funnel.io. 2) Normalization — map metrics to standard names: Spend, Impressions, Clicks, Conversions, Revenue, CPA, ROAS. Map Google's cost_micros (÷1M), Amazon's ACOS (convert to ROAS), Meta's Cost Per Result (= CPA). 3) Storage — load into BigQuery, Snowflake, or PostgreSQL. 4) Deduplication — use your analytics platform (GA4) as the source of truth. UTM parameters identify which platform drove each conversion. 5) Dashboard — Looker Studio, Tableau, or Google Sheets. Include: platform comparison table, blended totals, channel mix, trend charts, and CPA/ROAS by platform. 6) Add columns: Platform, Campaign, Spend, Impressions, Clicks, CTR, Conversions, CPA, ROAS, Impression Share (where available)."},

    {"user": "What is the difference between Google Merchant Center and Amazon product feeds?",
     "assistant": "Both are product catalogs that power shopping ads, but differ significantly. Google Merchant Center: feeds Shopping and PMax campaigns, requires standard attributes (id, title, description, link, image_link, price, availability, brand, gtin/mpn, condition), optimized titles boost performance, supplemental feeds allow quick updates, Google validates feed quality and disapproves non-compliant products. Content API v2.1 for programmatic management. Amazon Product Catalog: products are listed on Amazon itself (not just ads), catalog info comes from your Seller Central or Vendor Central listings, ad targeting uses ASINs and keywords, product detail pages are the landing pages. Amazon Attribution tracks off-Amazon traffic to Amazon pages. Key difference: Google sends traffic to YOUR website. Amazon keeps traffic on Amazon."},

    {"user": "Explain programmatic advertising and the role of DSPs, SSPs, and ad exchanges.",
     "assistant": "Programmatic advertising automates ad buying/selling through technology. The ecosystem: DSPs (Demand-Side Platforms) — advertisers use DSPs to buy ad inventory. Examples: The Trade Desk, DV360, Amazon DSP, Xandr. DSPs bid on inventory in real-time based on targeting criteria. SSPs (Supply-Side Platforms) — publishers use SSPs to sell their ad inventory. Examples: Google Ad Manager, Magnite, PubMatic, Index Exchange. SSPs connect to multiple ad exchanges to maximize revenue. Ad Exchanges — marketplaces where DSPs and SSPs meet. Google Ad Exchange is the largest. Open RTB auctions happen in under 100ms. DMPs (Data Management Platforms) — aggregate audience data for targeting. Increasingly replaced by CDPs (Customer Data Platforms) with first-party data. The flow: user loads a webpage → SSP sends bid request to ad exchange → DSPs evaluate and bid → highest bidder wins → ad is served — all in milliseconds."},
]

# Extend PLATFORM_PRETRAIN_TEXTS with mined repo content
_MINED_PRETRAIN = [
    # ═══════════════════════════════════════════════════════════
    # MINED FROM USER'S OWN REPOS — PRACTITIONER EXPERTISE
    # (Skills, Hub, Agent Specs, Blog)
    # ═══════════════════════════════════════════════════════════

    # From google-ads-math SKILL.md
    "PPC core formulas: CPA = Cost / Conversions. ROAS = Revenue / Cost. CTR = (Clicks / Impressions) * 100. CPC = Cost / Clicks. CVR = (Conversions / Clicks) * 100. CPM = (Cost / Impressions) * 1000. Budget projection: Monthly = (MTD Spend / Days Elapsed) * Days in Month. Break-even ROAS = 1 / Profit Margin.",
    "PPC benchmarks by campaign type: Search CTR 3-5%, CVR 3-8%, CPC $1-15. Shopping CTR 1-2%, CVR 1-3%, CPC $0.30-1.50. Display CTR 0.5-1%, CVR 0.5-1%, CPC $0.30-2. YouTube CTR 0.5-2%, CVR 0.5-1%, CPC $0.05-0.30. Industry CPA benchmarks: B2B SaaS $50-200, Ecommerce $10-50, Lead Gen $20-100.",
    "Impression share math: Impression Share = Impressions / Eligible Impressions. Missed impressions = (Impressions / IS) - Impressions. Cost to capture missed IS = Missed Impressions * (Current Cost / Current Impressions). Budget lost IS above 10% means budget constraint. Rank lost IS above 10% means Quality Score or bid issue.",

    # From google-ads-audit SKILL.md — 7-dimension framework
    "Google Ads 7-dimension audit framework: Dimension 1 Campaign Structure (15%): naming conventions, type appropriateness, ad group tightness under 20 keywords, match type strategy. Dimension 2 Keyword Strategy (20%): keyword relevance, match type distribution, negative keyword coverage, Quality Score distribution. Dimension 3 Ad Copy Quality (10%): RSA pin usage, ad strength scores, keyword insertion, extension coverage.",
    "Google Ads audit dimensions continued: Dimension 4 Bidding Strategy (15%): strategy appropriateness for conversion volume, target alignment with business goals, learning period status. Dimension 5 Budget Allocation (15%): budget lost impression share, budget distribution vs performance, shared budget issues, daily vs monthly pacing. Dimension 6 Conversion Tracking (15%): primary vs secondary actions, data freshness, attribution model, tag verification. Dimension 7 Audiences (10%): geographic precision, device adjustments, remarketing list sizes above 1000, remarketing coverage.",
    "Account audit scoring: overall score out of 100 with grade A through F. Critical issues require immediate fixes. 30-day action plan for quick wins. 60-day plan for structural improvements. 90-day plan for strategic optimizations. Each dimension scored individually with severity indicators.",

    # From google-ads-analysis SKILL.md — analysis framework
    "Analysis question classification: Performance check (how are my campaigns doing) requires pulling top-level metrics and comparing to benchmarks. Diagnostic questions (why is CPA increasing) require segmenting by dimension like time, device, geo, and keyword. Opportunity questions (where can I improve) require impression share, wasted spend, and Quality Score distribution analysis. Anomaly questions (something looks wrong) require comparing current vs prior period to find the break point.",
    "Analysis rules: always divide cost_micros by 1,000,000 for actual currency. Never recommend broad match without data to support it. Always check search terms before recommending budget increases. Always surface impression share as the number one missed metric. Every recommendation must include what to do, why it matters with data, expected quantified impact, and how to measure success with metric and timeline.",

    # From conversion-actions.md — the most misunderstood concept
    "Conversion action hierarchy is the most misunderstood concept in Google Ads. Primary conversions are used for Smart Bidding optimization. Secondary conversions are tracked for reporting only. The common mistake: setting every action as primary (page views, button clicks, form starts, form completions, purchases all optimizing simultaneously). The algorithm cannot optimize for contradictory goals. Fix: one primary conversion action per campaign objective, everything else secondary.",
    "Conversion action settings that matter: attribution model should be data-driven (default and recommended). Click-through window is 30 days default, shorten for impulse purchases, lengthen for B2B. View-through window is 1 day default, consider disabling for incrementality. Count setting: use One for leads (one form submit equals one lead), use Every for purchases (one visitor can buy multiple times).",

    # From enhanced-conversions.md — gotchas nobody documents
    "Enhanced conversions gotchas: Enhanced conversions for leads vs web are two different features. For web hashes user data from your site. For leads matches offline conversion imports. Most advertisers need both. You must accept customer data terms in Google Ads UI before the API works. SHA-256 hashing must happen before sending. Email normalization matters: lowercase, trim whitespace, remove dots from Gmail usernames. Different cases produce different hashes.",

    # From Cross-Platform-Terms.md — terminology mapping
    "Cross-platform account hierarchy: Google Ads uses MCC > Customer ID > Campaign > Ad Group > Ad. Meta Ads uses Business Manager > Ad Account > Campaign > Ad Set > Ad. Microsoft Ads uses Manager Account > Account > Campaign > Ad Group > Ad. Amazon Ads uses Portfolio > Advertiser > Campaign > Ad Group > Ad. The Trade Desk uses Partner > Advertiser > Campaign > Ad Group > Creative.",
    "Cross-platform metrics mapping: Google cost_micros divided by 1M equals dollars. Meta spend is already in dollars. Microsoft Spend is already in dollars. Amazon cost is already in dollars. Google clicks equals Meta link_clicks equals Microsoft Clicks equals Amazon clicks. Google conversions equals Meta actions equals Microsoft Conversions equals Amazon purchases.",
    "Cross-platform bidding strategy mapping: Google MAXIMIZE_CLICKS equals Meta LOWEST_COST_WITHOUT_CAP equals Microsoft MaxClicks. Google TARGET_CPA equals Meta COST_CAP equals Microsoft TargetCpa. Google TARGET_ROAS equals Meta MINIMUM_ROAS equals Microsoft TargetRoas. Google MANUAL_CPC equals Meta BID_CAP equals Microsoft ManualCpc.",
    "Cross-platform campaign types: Search campaigns exist on Google and Microsoft but not Meta or Amazon. Shopping exists as Google Shopping/PMax, Meta Catalog Sales, Microsoft Shopping, Amazon Sponsored Products. Display exists on Google, Microsoft Audience campaigns, Amazon Sponsored Display. Auto-optimized: Google Performance Max, Meta Advantage+, Microsoft Performance Max, Amazon Auto campaigns.",

    # From PPC Strategist agent — success metrics
    "PPC strategist success metrics: ROAS and CPA targets within 2 standard deviations. Brand impression share should be 90%+, non-brand top targets 40-60% budget permitting. Quality Score distribution: 70%+ of spend on QS 7+ keywords. Budget utilization 95-100% daily pacing with no more than 5% waste. Conversion volume growth target 15-25% quarter over quarter at stable efficiency. Testing velocity: 2-4 structured tests per month per account. New campaigns should reach steady state within 2-3 weeks.",

    # From GAQL Reference — complete patterns
    "GAQL common resources: campaign for campaign-level data, ad_group for ad group level, ad_group_ad for ad level, ad_group_criterion for keyword and targeting criteria, keyword_view for keyword performance joined with metrics, search_term_view for actual search queries that triggered ads, shopping_performance_view for product-level Shopping data, auction_insights for competitive metrics, change_event for account change history.",
    "GAQL key metric notes: cost_micros requires dividing by 1,000,000 for actual currency. average_cpc is already in micros, also divide by 1,000,000. conversions can be fractional due to data-driven attribution. search_impression_share is 0.0 to 1.0, multiply by 100 for percentage. When using BETWEEN or DURING date predicates, segments.date must be in the SELECT clause.",
]

PLATFORM_PRETRAIN_TEXTS.extend(_MINED_PRETRAIN)


PLATFORM_DPO_PAIRS = [
    {"prompt": "Should I run ads on every platform at once?",
     "chosen": "No. Start with 2-3 platforms where your audience is most active and your budget can generate statistically significant data (at least 30+ conversions/month per platform). For B2B: Google Search + LinkedIn. For ecommerce: Google Shopping + Meta. For apps: Google App + Meta + TikTok. Prove ROI on core platforms, then expand methodically. Spreading budget across too many platforms means none get enough data to optimize.",
     "rejected": "Yes, be everywhere! The more platforms you advertise on, the more people you reach. Set up campaigns on Google, Meta, LinkedIn, TikTok, Snapchat, Pinterest, Reddit, and Twitter simultaneously. Cast the widest net possible."},

    {"prompt": "How should I handle attribution across multiple ad platforms?",
     "chosen": "Use your analytics platform (GA4) as the single source of truth, not individual ad platform reports. Each platform takes credit for conversions it influenced, leading to over-counting. Set up consistent UTM parameters across all platforms. Implement server-side tracking for accuracy. Compare GA4 attributed conversions to platform-reported conversions to understand each platform's self-attribution inflation. Accept that no attribution model is perfect.",
     "rejected": "Just add up the conversions each platform reports. Google says 50 conversions, Meta says 30, LinkedIn says 20 — that means you got 100 total conversions. Each platform tracks independently so the numbers are accurate."},

    {"prompt": "Should I use a DSP or stick with platform-specific ad buying?",
     "chosen": "It depends on scale and goals. Use platform-specific buying (Google Ads, Meta Ads Manager) when: budget is under $10K/month, you need platform-specific features (Search, Shopping, Lead Gen Forms), or you're just starting. Use a DSP (The Trade Desk, DV360) when: budget exceeds $25K+/month, you need cross-channel programmatic reach (display, CTV, audio, DOOH), you want unified frequency management across publishers, or you need advanced audience data. Most advertisers should master platform-specific buying first, then add a DSP for incremental reach.",
     "rejected": "Always use a DSP — they're more sophisticated than platform-native tools. Direct platform buying is outdated. Move everything to programmatic for better performance across the board."},
]



def ________2():pass
"""
Training data mined from googleadsagent.ai blog articles.
Practitioner-level insights from 15+ years managing $48M+ annual ad spend.
"""
BLOG_PRETRAIN_TEXTS = [
    "Google's campaign-type documentation blurs objectives across types with near-identical copy. The practitioner's framework: if you know what people type before they buy, start with Search to capture existing high-intent demand. Layer Performance Max only after Search is already converting and you want incremental reach. PMax trades control for inventory breadth.",
    "Do not launch Performance Max without working conversion tracking and roughly 30 days of Search data to train the system. Launching cold wastes budget and learning. Demand Gen fits strong visual creative for awareness on YouTube, Discover, and Gmail. Display is for remarketing and cheap reach. Smart Campaigns are for tiny accounts with no manager.",
    "Enhanced CPC was deprecated for Search and Display in March 2025. Smart Bidding beats humans when conversion data is clean, values are correct, and structure is stable. Failure modes: broken tracking, low conversion volume, counting junk actions as conversions, and change churn triggering 5-7 days of learning volatility with each edit.",
    "Practical Smart Bidding sequencing: start with Maximize Conversions without a tight target to learn real achievable CPA, then add a target. Brand campaigns with consistent conversions are usually Smart Bidding. Non-brand with fewer than 30 monthly conversions may need Maximize Clicks first to build volume before switching to conversion-based bidding.",
    "Across 15 years of audits, roughly 90% of Google Ads performance problems trace to broken or incomplete conversion tracking, not copy, keywords, or budget. If Smart Bidding cannot see a trustworthy conversion, every automated dollar optimizes on noise.",
    "Google Tag firing conversions directly to Google Ads is the most reliable feed for Smart Bidding. GA4-imported conversions can work but introduce modeling and attribution differences. Enhanced conversions with hashed first-party data materially improve match quality where cookies fail, notably iOS.",
    "Only Primary conversion actions drive optimization. If micro-events like button clicks are marked Primary, the algorithm chases the wrong goal. Demote non-revenue actions to Secondary. One primary conversion action per campaign objective.",
    "Google's consolidation story is partly right: tiny ad groups with a few clicks a day starve Smart Bidding of data. But at $50K+ monthly spend, granular structure still matters: separate brand vs non-brand budgets, category-level campaigns for budget isolation, and geographic splits.",
    "Ad Strength is a directional creative signal, not a performance KPI. Excellent RSAs can lose to Poor ads when one headline perfectly fits the audience. The 2026 skill: automate bidding but own budgets, use broad match but maintain negatives, use RSAs but pin proven headlines.",
    "Daily budget is an average, not a hard cap. Google can spend up to 2x daily budget on a given day while averaging out over the month, bounded around 30.4x daily budget monthly. Mid-month budget changes recalculate the monthly limit, creating unexpected over or underspend.",
    "For strict monthly caps: use external pacing scripts projecting end-of-month spend, shared budgets across aligned campaigns sized from remaining monthly budget divided by days left, revised weekly. Changing budget mid-month is the most common source of pacing surprises.",
    "The real moat in Google Ads is not keywords or copy but measurement infrastructure. Accounts with clean Enhanced Conversions, proper Primary/Secondary designation, and server-side tracking consistently outperform accounts with double the budget but broken tracking.",
    "Overly aggressive Target CPA or Target ROAS throttles spend to near zero because the system avoids blowing the cap. Start with no target, observe the natural CPA, then set targets 10-20% above historical to avoid choking volume. Reduce targets gradually in 10-15% increments every 2 weeks.",
    "When evaluating campaign types: Search captures existing demand (bottom funnel), PMax extends reach across all channels (mid+bottom funnel), Demand Gen creates new demand with visual formats (top funnel), Display is cheap awareness and remarketing, Video is brand building unless using Video Action with conversion signal.",
    "The practitioner's weekly cadence: Monday review weekend performance and adjust bids. Wednesday review search terms and add negatives. Friday review pacing and prepare for weekend traffic patterns. Monthly: full performance review, QS audit, competitive analysis via Auction Insights.",
]

BLOG_SFT_PAIRS = [
    {"user": "Should we start with Performance Max or Search campaigns?",
     "assistant": "Start with Search if you know the queries people use before buying. Search captures existing demand and gives the most direct control over intent. Add Performance Max only after Search is converting reliably and you want incremental volume. Only launch PMax with solid conversion tracking plus at least 30 days of Search data. PMax spans all Google inventory in one campaign, so you trade fine-grained control for reach — launching it first without trained signal usually wastes spend."},

    {"user": "Google keeps recommending we consolidate campaigns. We spend over $50K/month — should we merge everything?",
     "assistant": "Partially follow Google. Consolidation helps when fragmentation starves algorithms — many ad groups each getting a handful of clicks. But at $50K+ monthly spend, keep what must be governed separately: brand vs non-brand budgets, high-value category campaigns, and geographic splits. Ad Strength is a hint, not proof — don't rewrite working ads just to chase an Excellent label. Pin winning assets instead of chasing the score. The 2026 skill is selective control: automate bidding but own budgets, use broad match but maintain negatives."},

    {"user": "Smart Bidding spends but we barely get conversions. What usually causes that?",
     "assistant": "Classic bad or thin signal: broken tags, too few conversions (need 30+ monthly), junk conversions like page views marked as Primary, or constant structural edits keeping campaigns in 5-7 day learning volatility. Unrealistic Target CPA or ROAS can also make the system pull back to near-zero conversions to avoid violating a cap. Fix: verify measurement first, aim for meaningful monthly volume, start with Maximize Conversions to learn a real baseline CPA, then constrain with targets. For very low-volume non-brand, use Maximize Clicks temporarily to build data."},

    {"user": "Why would imported GA4 conversions differ from Google Ads conversions for bidding?",
     "assistant": "They are not interchangeable. Google Tag events sent directly to Google Ads are the most dependable input for Smart Bidding. GA4 imports introduce modeling and attribution differences versus direct tagging. Additionally, Enhanced Conversions with hashed first-party data are critical because cookie loss (especially iOS) skews attribution — without them, CPA looks wrong and bidding optimizes the wrong reality. Only Primary conversion actions steer bidding. Secondary actions are for reporting. Mislabeling micro-conversions as Primary hijacks the optimizer."},

    {"user": "Google spent double my daily budget yesterday. Is that allowed and how do we cap a strict monthly number?",
     "assistant": "Yes — daily budget is an average, not a ceiling. Google can spend up to about 2x the daily amount on high-opportunity days while staying within roughly 30.4x daily budget over the month. Mid-month budget changes recalculate the monthly envelope, creating unexpected over or underspend. For hard monthly caps: use pacing scripts that project end-of-month spend with alerts. Use shared budgets sized from remaining monthly budget divided by days left, revised weekly. Never change budget mid-month without understanding the pacing implications."},
]


def ____():pass
def download_tokenizer():
    """Download minimind's 6400-vocab tokenizer from HuggingFace."""
    tokenizer_dir = "./model/tokenizer"
    if os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
        print(f"  Tokenizer already exists at {tokenizer_dir}")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download

    os.makedirs(tokenizer_dir, exist_ok=True)

    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        print(f"  Downloading {fname}...")
        hf_hub_download(
            "jingyaogong/MiniMind2", fname,
            local_dir=tokenizer_dir, local_dir_use_symlinks=False
        )

    print(f"✅ Tokenizer: {tokenizer_dir} (vocab_size=6400)")


def download_base_model(dim: int = 512):
    """Download minimind's pretrained checkpoint for transfer learning (skip pretrain)."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download

    out_dir = "./minimind_base"
    os.makedirs(out_dir, exist_ok=True)

    for fname in ["model.safetensors", "config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json"]:
        print(f"  Downloading {fname}...")
        hf_hub_download(
            "jingyaogong/MiniMind2", fname,
            local_dir=out_dir, local_dir_use_symlinks=False
        )

    print(f"✅ Base model: {out_dir}")
    print(f"   Use: python trainer/sft.py --load_from {out_dir} --from_transformers 1")

def ______():pass
def create_pretrain_data():
    """Generate advertising pretrain corpus — diverse knowledge chunks covering 60+ platforms."""
    path = "./dataset/pretrain_ads.jsonl"##===================================
    os.makedirs("./dataset", exist_ok=True)
    texts = [
        # === Google Ads Fundamentals ===
        "Google Ads uses a real-time auction system. When a user searches, Google runs an auction among all ads targeting that query. Ad Rank determines position and is calculated as Max CPC Bid multiplied by Quality Score, plus the expected impact of ad extensions. Higher Ad Rank means better position and often lower actual CPC.",
        "Quality Score in Google Ads is rated 1-10 for each keyword. It has three components: Expected Click-Through Rate (how likely users are to click your ad), Ad Relevance (how closely your ad matches the search intent), and Landing Page Experience (how relevant and useful your landing page is). A Quality Score of 7 or above is considered good.",
        "Google Ads campaign types include Search (text ads on search results), Display (banner ads across websites), Shopping (product listings), Video (YouTube ads), Performance Max (AI-optimized across all channels), App (mobile app installs), Demand Gen (social-style ads on Discover and YouTube), and Local (for physical stores).",
        "Cost Per Click (CPC) in Google Ads is determined by the auction. Actual CPC equals the Ad Rank of the advertiser below you divided by your Quality Score, plus one cent. This means higher Quality Scores directly reduce your costs. Average CPCs vary dramatically by industry: legal ($6-9), insurance ($15-50), B2B SaaS ($3-8), ecommerce ($1-3).",
        "Google Ads account structure follows a hierarchy: Account > Campaign > Ad Group > Ads and Keywords. Each campaign has its own budget and targeting settings. Ad groups contain related keywords and ads. Best practice: 15-20 keywords per ad group maximum, all closely themed. This is called Single Theme Ad Groups (STAGs).",

        # === GAQL ===
        "GAQL (Google Ads Query Language) queries follow the pattern: SELECT fields FROM resource WHERE conditions ORDER BY field LIMIT n. Resources include campaign, ad_group, ad_group_ad, ad_group_criterion, keyword_view, search_term_view, customer, and many more. Fields use dot notation like campaign.name or metrics.impressions.",
        "Common GAQL metrics include metrics.impressions, metrics.clicks, metrics.cost_micros (currency in millionths), metrics.conversions, metrics.conversions_value, metrics.search_impression_share, metrics.search_budget_lost_impression_share, metrics.search_rank_lost_impression_share, and metrics.all_conversions.",
        "GAQL date filtering uses segments.date with special date ranges: DURING TODAY, YESTERDAY, LAST_7_DAYS, LAST_14_DAYS, LAST_30_DAYS, THIS_MONTH, LAST_MONTH, THIS_QUARTER, LAST_QUARTER, THIS_YEAR, LAST_YEAR. Custom ranges use BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'.",
        "GAQL supports filtering with WHERE clauses. Common operators: = (equals), != (not equals), > < >= <= (comparisons), IN (list membership), LIKE (pattern matching), CONTAINS ANY/ALL (array operations), IS NULL, IS NOT NULL, DURING (date ranges). Multiple conditions use AND.",
        "The search_term_view resource in GAQL shows what users actually searched. Key query: SELECT search_term_view.search_term, metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions FROM search_term_view WHERE segments.date DURING LAST_30_DAYS ORDER BY metrics.cost_micros DESC. This reveals wasted spend on irrelevant queries.",

        # === Bid Strategies ===
        "Google Ads Smart Bidding strategies use machine learning. Target CPA automatically sets bids to get conversions at your target cost-per-acquisition. Target ROAS sets bids to maximize conversion value at your target return-on-ad-spend. Maximize Conversions gets the most conversions within budget. Maximize Conversion Value gets the most revenue within budget.",
        "Manual CPC bidding gives full control over keyword-level bids. Enhanced CPC (ECPC) adjusts manual bids up or down based on conversion likelihood. Smart Bidding strategies need at least 30 conversions in 30 days to optimize effectively. New campaigns should start with Maximize Clicks or manual CPC until conversion data accumulates.",
        "Target CPA bidding works best with at least 50 conversions per month per campaign. Set the initial target at or slightly above your historical CPA to avoid throttling. Smart Bidding has a learning period of approximately 7-14 days after any change. Avoid making changes during this period or performance data will be unreliable.",
        "Target ROAS bidding requires at least 15 conversions in 30 days per campaign and works best with revenue tracking. Initial target should be set at or below historical ROAS. A target ROAS of 400% means you expect $4 in revenue for every $1 spent. ROAS targets that are too aggressive will severely limit impression volume.",

        # === Match Types ===
        "Keyword match types control which searches trigger your ads. Exact match [keyword] shows ads for searches that match the meaning of your keyword. Phrase match \"keyword\" shows ads for searches that include the meaning of your keyword. Broad match keyword shows ads for searches related to your keyword. Broad match modifier was sunset and merged into phrase match.",
        "Negative keywords prevent ads from showing for irrelevant searches. They also have match types: negative broad (default), negative phrase, and negative exact. Negative broad match excludes searches containing all negative keyword terms in any order. Negative phrase excludes searches containing the exact phrase. Regular review of the Search Terms Report to add negatives is critical for account health.",

        # === Conversion Tracking ===
        "Google Ads conversion tracking uses the global site tag (gtag.js) or Google Tag Manager. Primary conversions are used by Smart Bidding for optimization. Secondary conversions are observed but not used for bidding. Enhanced conversions improve measurement accuracy by securely sending hashed first-party data like email addresses.",
        "Offline conversion tracking imports conversion data back to Google Ads using Google Click ID (gclid) or Enhanced Conversions for Leads. This is critical for B2B where the conversion (e.g., closed deal) happens days or weeks after the click. Upload frequency should be daily or at minimum weekly to give Smart Bidding fresh signals.",
        "Attribution models determine how credit for conversions is assigned. Last click gives 100% credit to the last-clicked ad. Data-driven attribution uses machine learning to distribute credit across touchpoints. Time decay gives more credit to interactions closer to conversion. Google deprecated First Click, Linear, and Position-Based models in favor of Data-Driven Attribution.",

        # === Performance Max ===
        "Performance Max (PMax) campaigns use AI to optimize across all Google channels: Search, Shopping, Display, YouTube, Discover, Gmail, and Maps. They require Asset Groups containing text headlines (up to 15), long headlines (up to 5), descriptions (up to 5), images (up to 20), videos (up to 5), and audience signals.",
        "Performance Max campaigns can cannibalize brand search traffic. Monitor your brand campaign impression share before and after launching PMax. Use brand exclusions (available in PMax settings) to prevent PMax from bidding on your brand terms. PMax should complement, not replace, dedicated brand and non-brand search campaigns.",
        "Asset performance in Performance Max is rated Low, Good, or Best. Replace Low-performing assets regularly. PMax reporting is limited — use Insights tab and asset group reports to understand performance. Third-party scripts can extract more detailed PMax data via the Google Ads API.",

        # === Shopping Campaigns ===
        "Google Shopping campaigns require a Merchant Center feed. Product data must include id, title, description, link, image_link, price, availability, brand, gtin or mpn, and condition. Feed quality directly impacts Shopping ad performance. Optimize titles with primary keywords, brand, and key attributes.",
        "Shopping campaign priority settings (High, Medium, Low) control which campaign serves when multiple campaigns have products matching a search. Common strategy: High priority campaign with low bids for broad terms, Medium for mid-funnel, Low priority with highest bids for proven converters. Combined with shared negative keyword lists for query funneling.",

        # === Meta Ads ===
        "Meta Ads (Facebook and Instagram) use a campaign-ad set-ad hierarchy. Campaign level sets the objective: Awareness, Traffic, Engagement, Leads, App Promotion, or Sales. Campaign Budget Optimization (CBO) distributes budget across ad sets automatically. Ad set level controls targeting, placement, and schedule.",
        "Meta Ads targeting includes Core Audiences (demographics, interests, behaviors), Custom Audiences (website visitors via Pixel, customer lists, app users, video viewers), and Lookalike Audiences (users similar to your custom audiences). Lookalike audience size ranges from 1% (most similar) to 10% (broader reach).",
        "The Meta Pixel tracks website actions for conversion optimization and audience building. Standard events include PageView, ViewContent, AddToCart, InitiateCheckout, Purchase, Lead, CompleteRegistration, and Search. The Conversions API (CAPI) sends server-side events directly to Meta, improving tracking accuracy and attribution.",
        "Meta Ads attribution uses a 7-day click, 1-day view default window. iOS 14.5+ privacy changes via App Tracking Transparency significantly impacted Meta's tracking and attribution. Aggregated Event Measurement limits optimization events to 8 per domain. Server-side Conversions API integration helps recover lost attribution data.",
        "Meta Advantage+ campaigns use AI automation similar to Google's Performance Max. Advantage+ Shopping campaigns require minimal setup: just a catalog, creative assets, and a budget. They automatically optimize targeting, placement, and creative combinations. Best for advertisers with established pixel data and product catalogs.",

        # === Microsoft Ads ===
        "Microsoft Advertising (formerly Bing Ads) offers lower average CPCs than Google Ads, typically 30-50% less. The audience skews older and higher-income. Key differentiator: LinkedIn profile targeting allows targeting by company, job function, and industry. Import campaigns directly from Google Ads to quickly launch on Microsoft.",
        "Universal Event Tracking (UET) is Microsoft Ads' equivalent of Google's conversion tag. It tracks page visits, custom events, and revenue. Microsoft Ads supports automated bidding strategies similar to Google: Target CPA, Target ROAS, Maximize Conversions, and Enhanced CPC.",

        # === Amazon Ads ===
        "Amazon Advertising includes Sponsored Products (keyword-targeted product listings), Sponsored Brands (banner ads with brand logo and multiple products), and Sponsored Display (audience and product targeting across Amazon and third-party sites). ACOS (Advertising Cost of Sales) equals ad spend divided by ad revenue, the primary efficiency metric.",
        "Amazon Sponsored Products use automatic targeting (Amazon matches to relevant searches) and manual targeting (you choose keywords or products). Automatic campaigns are great for keyword research — let them run for 2 weeks, then mine the Search Term Report for converting queries to add to manual campaigns.",

        # === PPC Math ===
        "Cost Per Acquisition (CPA) = Total Cost / Total Conversions. Return on Ad Spend (ROAS) = Conversion Value / Cost. Click-Through Rate (CTR) = Clicks / Impressions. Conversion Rate (CVR) = Conversions / Clicks. Cost Per Mille (CPM) = (Cost / Impressions) × 1000. These are the fundamental PPC metrics.",
        "Break-even ROAS = 1 / Profit Margin. If your profit margin is 25%, break-even ROAS = 400% (you need $4 revenue per $1 ad spend). Target ROAS should exceed break-even ROAS. For a 40% margin business, break-even ROAS is 250%, so targeting 350-400% provides healthy profit.",
        "Impression Share = Impressions Received / Total Eligible Impressions. Lost Impression Share (Budget) = impressions lost because daily budget was exhausted. Lost Impression Share (Rank) = impressions lost because Ad Rank was too low (bid or Quality Score). If Budget Lost IS > 10%, consider increasing budget or narrowing targeting.",
        "Customer Lifetime Value (CLV) should inform your CPA targets. If CLV is $500 and you want 3x return, your max CPA is $167. Most advertisers set CPA targets based only on the first transaction, leaving money on the table. Calculate CLV = (Average Order Value × Purchase Frequency × Customer Lifespan) and set CPA accordingly.",
        "Click-weighted average CPC = Total Cost / Total Clicks. However, different keywords have vastly different CPCs within the same campaign. Segmenting by match type typically shows: Exact < Phrase < Broad in CPC but Exact > Phrase > Broad in conversion rate. The most efficient strategy balances reach and precision.",

        # === Audiences ===
        "Remarketing lists let you re-engage users who previously visited your website. Segment by recency (1-7 days, 8-30 days, 31-90 days) and behavior (viewed product, added to cart, visited pricing page). Cart abandoner audiences typically have 3-5x higher conversion rates than cold traffic. Minimum list size is 1,000 users for Search and 100 for Display.",
        "Customer Match in Google Ads lets you upload customer email lists for targeting. Match rates typically range from 30-60%. Use Customer Match for upselling, cross-selling, and exclusion (exclude existing customers from acquisition campaigns). Combined with Similar Audiences, it extends reach to lookalike prospects.",
        "In-Market audiences target users actively researching or comparing products in specific categories. Google identifies these users through search behavior and content consumption patterns. In-Market audiences are available for both Search and Display campaigns. They work well as observation audiences to gather data before targeting.",

        # === Ad Copy and Creative ===
        "Responsive Search Ads (RSAs) allow up to 15 headlines (30 chars each) and 4 descriptions (90 chars each). Google's AI tests combinations and learns which perform best. Best practices: include keywords in 3+ headlines, add unique value propositions, use numbers and CTAs, and pin critical messages to ensure they always show.",
        "Ad copy best practices: lead with the benefit not the feature, include a clear call to action, use the target keyword in headline 1, differentiate from competitors, include social proof (ratings, awards, numbers), and create urgency without being misleading. Test 2-3 RSA variants per ad group.",

        # === Landing Pages ===
        "Landing page experience is a Quality Score component. Key factors: relevance to ad and keyword, page load speed (under 3 seconds), mobile responsiveness, clear above-the-fold value proposition, prominent CTA, trust signals (testimonials, security badges, logos), and minimal distracting navigation.",
        "Landing page conversion rate optimization: test one element at a time, start with the headline and CTA, remove navigation links that leak traffic, match the landing page message to the ad copy, use form field reduction (every field removed increases completion rates ~10%), and implement exit-intent offers for abandoning visitors.",

        # === Account Optimization ===
        "The 80/20 rule applies to Google Ads: approximately 80% of conversions come from 20% of keywords. Identify your top-performing keywords and ensure they have maximum impression share. Simultaneously, the bottom 20% of keywords by performance often waste significant budget — pause or reduce bids on these.",
        "Search Terms Report review should happen weekly. Look for: irrelevant queries consuming budget (add as negatives), high-converting queries not yet added as keywords (add them), close variants triggering unexpected matches (tighten match types or add negatives), and new market trends appearing in user searches.",
        "Account optimization cadence: daily — check budget pacing and anomalies. Weekly — review search terms, adjust bids, check Quality Score changes. Monthly — full performance review, add/pause keywords, refresh ad copy. Quarterly — strategy review, budget reallocation, landing page testing, competitive analysis.",

        # === Cross-Platform Strategy ===
        "Cross-platform advertising strategy should follow the customer journey. Google Search captures bottom-funnel demand (users actively searching). Meta and TikTok create top-funnel awareness (users discovering). LinkedIn targets B2B decision-makers. YouTube builds mid-funnel consideration. Retarget across all platforms for maximum conversion.",
        "Budget allocation across platforms depends on business model. B2B services: 50% Google Search, 20% LinkedIn, 15% Meta, 15% Display/YouTube. D2C ecommerce: 40% Google (Search + Shopping), 35% Meta/Instagram, 15% TikTok, 10% other. Lead gen: 60% Google Search, 25% Meta, 15% Microsoft. Adjust based on actual CPL/ROAS data.",
        "When comparing metrics across platforms, normalize terminology. Google's CPA is Meta's Cost Per Result. Google's ROAS is Amazon's inverse ACOS (ROAS = 1/ACOS × 100). Google's Search Impression Share has no direct Meta equivalent. Create a unified reporting framework mapping each platform's metrics to standard KPIs.",

        # === Programmatic / DSP ===
        "Demand-Side Platforms (DSPs) like The TradeDesk, DV360, and Amazon DSP enable programmatic buying across display, video, native, audio, and CTV inventory. Real-Time Bidding (RTB) auctions happen in milliseconds. Programmatic excels at scale and audience precision but requires larger budgets ($5K+ monthly) to optimize effectively.",

        # === LinkedIn Ads ===
        "LinkedIn Ads are premium-priced (CPCs of $5-15+) but reach professional audiences with unmatched B2B targeting. Target by job title, company, industry, seniority, skills, and groups. Lead Gen Forms pre-populate with LinkedIn profile data, achieving 2-5x higher conversion rates than website landing pages for B2B lead gen.",

        # === Reddit Ads ===
        "Reddit Ads target communities (subreddits) with shared interests. Successful Reddit ads use authentic language — overly polished corporate messaging gets downvoted. Promoted posts appear in feeds. Interest targeting and conversation placement extend reach beyond specific subreddits. CPCs are typically low ($0.50-3) but engagement quality varies.",

        # === TikTok Ads ===
        "TikTok Ads use a campaign-ad group-ad hierarchy. The platform excels at reaching Gen Z and Millennials with short-form video. Spark Ads boost existing organic content as paid ads. TikTok Shop integrates ecommerce directly into the platform. Smart Performance Campaigns are TikTok's equivalent of PMax — fully automated.",

        # === Seasonality and Trends ===
        "CPCs in Google Ads fluctuate seasonally. Q4 (October-December) has the highest CPCs due to holiday advertising demand. January often sees lower CPCs as advertisers pull back. Industry-specific peaks matter too: tax software in February-April, travel in January and June, back-to-school in July-August. Plan budgets around these patterns.",
        "Auction Insights report shows how you compare to competitors in the same auctions. Key metrics: impression share (your share vs competitors), overlap rate (how often you appear in the same auctions), outranking share (how often your ad ranked higher), position above rate, and top of page rate. Monitor monthly for competitive trends.",

        # === API and Automation ===
        "The Google Ads API (v23) enables programmatic campaign management. Common automation tasks: bid adjustments based on weather or inventory data, automated search terms report processing, scheduled reporting, budget pacing alerts, Quality Score monitoring, and ad copy testing workflows. Rate limits: 15,000 requests per day per developer token.",
        "Google Ads Scripts use JavaScript to automate account tasks without API setup. Common scripts: automatic bid adjustments, broken URL detection, Quality Score tracking, search term mining for negatives, performance anomaly alerts, and automated reporting. Scripts run on a schedule and have a 30-minute execution time limit.",

        # === B2B Specific ===
        "B2B Google Ads strategy differs from B2C: longer sales cycles (30-180 days), multiple decision makers, higher CPAs but higher deal values. Use offline conversion tracking to feed closed-deal data back to Smart Bidding. Target by audience (in-market B2B segments) + keyword for precision. Bid higher during business hours when decision-makers are active.",
        "Lead quality scoring is essential for B2B PPC. Not all leads are equal — track cost per SQL (sales qualified lead), not just cost per lead. Feed lead quality data back via offline conversion imports with value-based bidding. This teaches Smart Bidding to optimize for revenue, not just form fills.",

        # === Ecommerce Specific ===
        "Ecommerce Google Ads strategy centers on Shopping + Performance Max. Optimize your Merchant Center feed: titles should include brand + product type + key attribute + color/size. Use Supplemental feeds for quick changes. Custom labels allow campaign segmentation by margin, price tier, bestseller status, or seasonality.",
        "ROAS calculation for ecommerce: Revenue / Ad Spend × 100. A ROAS of 500% means $5 revenue per $1 spend. But ROAS alone is insufficient — consider profit margin. $100 revenue at 500% ROAS with 20% margin = $20 profit - $20 ad spend = breakeven. Factor in margins, shipping costs, and returns for true profitability.",
    ]


    texts.extend(PLATFORM_PRETRAIN_TEXTS)##===================================
    texts.extend(BLOG_PRETRAIN_TEXTS)##===================================
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(150):
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"✅ Pretrain: {path} ({len(texts)} unique texts × 150 = {len(texts) * 150} samples)")


def create_sft_data():
    """Generate advertising SFT instruction-response pairs — 60+ platforms."""
    path = "./dataset/sft_ads.jsonl"##===================================
    os.makedirs("./dataset", exist_ok=True)
    pairs = [
        # === PPC Math ===
        {"user": "What is CPA and how do I calculate it?",
         "assistant": "CPA (Cost Per Acquisition) = Total Ad Spend / Total Conversions. If you spent $5,000 and got 100 conversions, your CPA is $50. For target CPA, work backwards from your unit economics: if your product profit margin is $200, a CPA under $200 is profitable. Most B2B SaaS targets CPA at 10-20% of first-year Annual Contract Value."},

        {"user": "How do I calculate ROAS and what's a good target?",
         "assistant": "ROAS (Return On Ad Spend) = Revenue from Ads / Ad Spend. If you spent $1,000 and generated $4,000 in revenue, your ROAS is 4x or 400%. Break-even ROAS = 1 / Profit Margin. For a 25% margin business, break-even is 400%. Your target should exceed break-even: aim for 500-600% to generate actual profit. For a 50% margin business, break-even is 200% so 300%+ is profitable."},

        {"user": "My CTR is 1.2%. Is that good or bad?",
         "assistant": "It depends on campaign type and industry. For Google Search ads, average CTR is 3-5% — so 1.2% is below average and signals your ads aren't resonating. Check: are your keywords too broad? Is your ad copy relevant to search intent? For Display ads, 0.5-1% is normal so 1.2% would be excellent. For Shopping, 1-2% is typical. Fix low Search CTR by improving headline relevance, adding keyword insertion, testing stronger CTAs, and tightening match types."},

        {"user": "Calculate impression share from these numbers: 50,000 impressions received, estimated 125,000 eligible impressions.",
         "assistant": "Search Impression Share = Impressions / Eligible Impressions = 50,000 / 125,000 = 40%. You're missing 60% of eligible auctions. Check the breakdown: if Budget Lost IS is 35%, your daily budget runs out early — increase budget or reduce bids on low-performers. If Rank Lost IS is 25%, your Ad Rank is too low — improve Quality Score or increase bids on high-value keywords."},

        # === GAQL Queries ===
        {"user": "Write a GAQL query to find my top 10 campaigns by spend last 30 days.",
         "assistant": "SELECT campaign.name, campaign.status, metrics.cost_micros, metrics.impressions, metrics.clicks, metrics.conversions, metrics.conversions_value, metrics.search_impression_share FROM campaign WHERE segments.date DURING LAST_30_DAYS AND campaign.status != 'REMOVED' ORDER BY metrics.cost_micros DESC LIMIT 10. Remember: cost_micros is in millionths — divide by 1,000,000 for actual currency. So cost_micros = 45000000 means $45.00."},

        {"user": "Write a GAQL query to find wasted spend — search terms with spend but zero conversions.",
         "assistant": "SELECT search_term_view.search_term, search_term_view.status, metrics.impressions, metrics.clicks, metrics.cost_micros, metrics.conversions FROM search_term_view WHERE segments.date DURING LAST_30_DAYS AND metrics.cost_micros > 0 AND metrics.conversions = 0 ORDER BY metrics.cost_micros DESC LIMIT 50. This shows your biggest money-wasters. Add irrelevant terms as negative keywords. In unoptimized accounts, 20-40% of spend typically goes to irrelevant search terms."},

        {"user": "Write a GAQL query to check Quality Score for all active keywords.",
         "assistant": "SELECT ad_group.name, ad_group_criterion.keyword.text, ad_group_criterion.keyword.match_type, ad_group_criterion.quality_info.quality_score, ad_group_criterion.quality_info.creative_quality_score, ad_group_criterion.quality_info.search_predicted_ctr, ad_group_criterion.quality_info.post_click_quality_score, metrics.impressions, metrics.cost_micros FROM ad_group_criterion WHERE ad_group_criterion.type = 'KEYWORD' AND ad_group_criterion.status = 'ENABLED' AND campaign.status = 'ENABLED' ORDER BY ad_group_criterion.quality_info.quality_score ASC."},

        {"user": "Write a GAQL query for Auction Insights to see my competitors.",
         "assistant": "SELECT auction_insights.display_url, auction_insights.search_impression_share, auction_insights.search_overlap_rate, auction_insights.search_outranking_share, auction_insights.search_top_impression_percentage FROM auction_insights WHERE segments.date DURING LAST_30_DAYS. This shows who you're competing against and how often they outrank you. If a new competitor suddenly appears with high impression share, they may be driving up your CPCs."},

        {"user": "Write a GAQL query to find ad groups with high spend but low conversion rate.",
         "assistant": "SELECT ad_group.name, campaign.name, metrics.cost_micros, metrics.clicks, metrics.conversions, metrics.impressions FROM ad_group WHERE segments.date DURING LAST_30_DAYS AND metrics.clicks > 50 AND metrics.conversions < 2 ORDER BY metrics.cost_micros DESC LIMIT 20. Ad groups with 50+ clicks and fewer than 2 conversions have under 4% CVR. Review their keywords, ad copy, and landing pages. Consider pausing the worst performers."},

        # === Diagnostic Questions ===
        {"user": "My CPA is $85 and my target is $50. What should I do?",
         "assistant": "CPA is 70% above target — here's the systematic fix: 1) Search Terms Report: find irrelevant queries eating budget, add as negatives. This alone often cuts CPA 15-30%. 2) Keyword performance: pause keywords with CPA over $100 that haven't improved in 30 days. 3) Landing page: check conversion rate — if under 3%, the page is the bottleneck. Test headline, form length, CTA. 4) Ad copy: ensure message match between keyword intent, ad text, and landing page. 5) Bid strategy: if using Target CPA, set it to $60 (not $50) and reduce gradually. Too aggressive a target throttles volume."},

        {"user": "My impression share dropped from 75% to 40% in one week. What happened?",
         "assistant": "A 35-point drop in one week is dramatic. Three possible causes in order of likelihood: 1) Budget constraint — check search_budget_lost_impression_share in GAQL. If your daily budget ran out by 2pm, you lost the rest of the day's auctions. Fix: increase budget or reduce bids/keywords. 2) Quality Score drop — check search_rank_lost_impression_share. Look for disapproved ads, landing page issues, or keyword QS changes. 3) New competitor — check Auction Insights for a new entrant with aggressive bids. For a sudden drop, budget exhaustion is the most common cause."},

        {"user": "My Google Ads account spends $10K/month but generates only 15 conversions. What's wrong?",
         "assistant": "At $10K spend and 15 conversions, your CPA is $667 — almost certainly too high unless you're selling high-ticket items. Systematic diagnosis: 1) Check conversion tracking first — are conversions actually being recorded? Test by completing a conversion yourself. 2) Run the Search Terms Report — at this spend level with few conversions, irrelevant traffic is likely consuming most budget. 3) Check landing page analytics separately — if you're getting clicks but no conversions, the page is the problem. 4) Look at keyword match types — broad match without Smart Bidding at this scale bleeds budget. 5) Verify targeting — check geographic, device, and audience settings for anything too broad."},

        {"user": "Should I use broad match or exact match keywords?",
         "assistant": "Both, strategically. Start with exact match for your highest-intent, proven keywords — this gives maximum control and typically highest conversion rates. After collecting 2 weeks of data, expand top performers to phrase match to capture more variations. Only use broad match when: you have Smart Bidding enabled (Target CPA or ROAS), you have robust negative keyword lists, and you have sufficient conversion volume (30+ per month). Broad match without these guardrails wastes budget on irrelevant queries. The progression is: Exact (control) → Phrase (expand) → Broad with Smart Bidding (scale)."},

        # === Campaign Structure ===
        {"user": "How should I structure Google Ads campaigns for a B2B SaaS company?",
         "assistant": "Recommended structure: Campaign 1 — Brand (10% budget): exact match brand terms, highest ROAS. Campaign 2 — High-Intent Non-Brand (40%): keywords like 'buy [product category]', '[competitor] alternative', '[solution] software'. Campaign 3 — Mid-Funnel (25%): problem-aware keywords like 'how to [solve problem]', '[pain point] solution'. Campaign 4 — Competitor (15%): competitor brand name targeting with comparison messaging. Campaign 5 — Remarketing (10%): Display/YouTube retargeting visitors who didn't convert. Start with Campaigns 1-2, prove ROI, then expand. Use offline conversion tracking to feed closed-deal data back for Smart Bidding optimization."},

        {"user": "How should I structure campaigns for an ecommerce store with 500 products?",
         "assistant": "Use a tiered structure based on product value: Campaign 1 — Brand Search (10-15% budget): exact match brand terms, protect your brand. Campaign 2 — Top Products via Performance Max (40-50%): your top 50 products by profit margin × sales volume. Use custom labels in Merchant Center to segment. Campaign 3 — Standard Shopping by Category (25-30%): organize by product category with priority settings for query funneling. Campaign 4 — Non-Brand Search (10-15%): high-intent category terms like 'buy [product type] online'. Campaign 5 — Remarketing (5-10%): cart abandoners and product page viewers. Start with Campaigns 1-2 and prove ROI before expanding."},

        # === Platform Comparisons ===
        {"user": "Compare Google Ads and Meta Ads for lead generation.",
         "assistant": "Google Ads Search: intent-based (users actively searching), higher CPC ($2-15), higher conversion rate (3-8%), best for bottom-funnel high-intent leads. Typical CPL: $30-100 B2B, $15-50 B2C. Meta Ads: interruption-based (users browsing social media), lower CPC ($0.50-3), lower conversion rate (1-3%), best for top-funnel awareness and visual products. Typical CPL: $15-60 B2B, $5-25 B2C. Use both: Google captures existing demand, Meta creates new demand. Start with 70/30 Google/Meta split and adjust based on 30 days of CPL data. Meta is better for retargeting warm audiences, Google is better for capturing purchase intent."},

        {"user": "When should I use Microsoft Ads vs Google Ads?",
         "assistant": "Use both. Microsoft Ads (Bing) reaches 36% of US desktop search. Key advantages: CPCs are 30-50% lower than Google, audience skews older and higher-income (good for B2B, financial services, luxury), and LinkedIn profile targeting enables B2B precision no other platform offers. Import your Google campaigns directly for a quick start. Allocate 10-20% of your Google Search budget to Microsoft. The lower competition often delivers better ROAS on the same keywords."},

        {"user": "Is Amazon Ads worth it for my products?",
         "assistant": "If you sell on Amazon, yes — Amazon Ads directly drives sales on the platform where purchase intent is highest. Start with Sponsored Products using automatic targeting to discover what converts. After 2 weeks, mine the Search Term Report and create manual campaigns with proven keywords. ACOS (Advertising Cost of Sales) = Ad Spend / Ad Revenue. Target ACOS should be below your profit margin. If your margin is 30%, keep ACOS under 25% to stay profitable. Sponsored Brands build brand awareness at the top of search results. Sponsored Display retargets shoppers who viewed your products."},

        # === Common Tasks ===
        {"user": "What negative keywords should I add to a B2B SaaS campaign?",
         "assistant": "Start with these universal B2B negatives: free, cheap, open source, download, tutorial, course, certification, salary, jobs, career, intern, reddit, youtube, how to, what is, DIY. Then add industry-specific negatives: for SaaS add self-hosted and on-premise (if cloud-only), for enterprise add small business and personal, for paid tools add free trial (unless you offer one). Review Search Terms Report weekly and add new negatives. Build a shared negative keyword list at account level to apply across all campaigns. Typically 200-500 negatives are needed for a well-optimized B2B account."},

        {"user": "How do I find and fix wasted ad spend?",
         "assistant": "Five areas to check: 1) Search Terms Report — filter for cost > $0 and conversions = 0, sorted by cost descending. Add irrelevant terms as negatives. 2) Geographic performance — check if certain locations consume budget without converting. Exclude or reduce bids for underperformers. 3) Device performance — mobile vs desktop conversion rates differ dramatically by industry. Adjust device bid modifiers. 4) Time-of-day — run a day-of-week and hour-of-day report. B2B often wastes money on weekends and nights. Use ad scheduling. 5) Low Quality Score keywords — keywords with QS 3 or below pay significantly more per click. Fix ad relevance or pause them."},

        {"user": "How do I set up conversion tracking in Google Ads?",
         "assistant": "Two methods: Option 1 — Google Tag Manager (recommended): create a conversion action in Google Ads, copy the conversion ID and label, create a Google Ads Conversion Tracking tag in GTM triggered on your thank-you page or form submission event. Option 2 — Global site tag: add the gtag.js snippet to every page, then add the event snippet on the conversion page. For both methods: set the conversion counting (one per click for leads, every for purchases), assign a value, and choose the right attribution model (data-driven recommended). Test with Google Tag Assistant. Then in Google Ads, mark this as a primary conversion so Smart Bidding uses it."},

        {"user": "How do I audit a Google Ads account?",
         "assistant": "7-dimension audit: 1) Conversion tracking — verify it works, check for duplicate counting, review attribution model. 2) Account structure — campaigns organized by theme, 15-20 keywords per ad group max, correct match types. 3) Keywords — review search terms for waste, check Quality Scores, identify gaps. 4) Ad copy — at least 2 RSAs per ad group, all ad strength 'Good' or better, extensions in use. 5) Bidding — appropriate strategy for conversion volume, targets realistic, learning period respected. 6) Budget — no campaigns limited by budget unintentionally, spend aligned with priority. 7) Landing pages — relevant, fast, mobile-friendly, conversion-optimized. Score each dimension 1-10 and prioritize fixes by impact."},

        # === Performance Max ===
        {"user": "Should I use Performance Max campaigns?",
         "assistant": "PMax works well when you have: 1) strong conversion data (50+ conversions/month), 2) quality creative assets across text, images, and video, 3) properly configured conversion tracking, 4) defined audience signals. Start PMax alongside existing Search campaigns, don't replace them. Watch for brand traffic cannibalization — use brand exclusions. PMax reporting is limited, so use the Insights tab. Don't launch PMax with a new account that has no conversion history — Smart Bidding needs data to optimize. For ecommerce, PMax with a product feed is especially powerful."},

        # === Bid Strategy ===
        {"user": "When should I switch from manual CPC to Smart Bidding?",
         "assistant": "Switch to Smart Bidding when you have: 1) at least 30 conversions in 30 days (50+ is ideal), 2) reliable conversion tracking with accurate values, 3) consistent campaign performance for the algorithm to learn from. Start with Maximize Conversions (no target) for 2 weeks, then add a Target CPA set at or above your historical CPA. Reduce the target gradually in 10-15% increments, waiting 2 weeks between changes. If using Target ROAS, you need at least 15 conversions in 30 days. Never change bid strategy during a promotional period or unusual traffic spike."},

        # === Budget ===
        {"user": "How much should I spend on Google Ads to start?",
         "assistant": "Work backwards from your target: if your target CPA is $50 and you want 30 leads per month, you need $1,500/month budget. But for testing, you need enough data to optimize — minimum $30-50/day per campaign, which means $900-1,500/month per campaign. With 2 campaigns (brand + non-brand), budget $2,000-3,000/month for a meaningful test. Run for at least 60 days before judging results — the first 30 days are learning. Never set a budget you can't sustain for 3 months. Too-small budgets spread across too many campaigns is the most common mistake."},

        # === Quality Score ===
        {"user": "How do I improve Quality Score from 4 to 7+?",
         "assistant": "Quality Score has three components — fix the weakest one first: 1) Expected CTR (Below Average → Average): improve ad headlines to be more specific and compelling, add keyword in headline 1, use ad customizers, test emotional triggers. 2) Ad Relevance (Below Average → Average): ensure your ad copy directly addresses the keyword intent, use keyword insertion where appropriate, create tighter ad groups with fewer, more related keywords. 3) Landing Page Experience (Below Average → Average): improve page load speed to under 3 seconds, ensure mobile responsiveness, match landing page content to the keyword/ad message, add relevant content above the fold, remove pop-ups. Expect 2-4 weeks for QS changes to take effect after improvements."},

        # === Reporting ===
        {"user": "What should a monthly Google Ads performance report include?",
         "assistant": "Essential sections: 1) Executive summary — spend, conversions, CPA, ROAS vs targets, month-over-month trend. 2) Campaign performance table — each campaign's spend, impressions, clicks, CTR, conversions, CPA, ROAS. 3) Top/bottom performers — best and worst keywords, ad groups, search terms. 4) Impression share analysis — total IS, budget lost IS, rank lost IS, competitive trends. 5) Quality Score distribution — how many keywords at each QS level, changes from prior month. 6) Audience insights — device, geography, demographics if available. 7) Recommendations — specific actions for next month with expected impact. 8) Budget recommendation — should budget increase, decrease, or shift between campaigns."},

        # === Ad Extensions ===
        {"user": "What ad extensions should I always use?",
         "assistant": "Always enable these: 1) Sitelink extensions — add 4-8 links to key pages (pricing, features, case studies, demo). 2) Callout extensions — 4+ highlights like 'Free Trial', '24/7 Support', 'No Setup Fee'. 3) Structured snippets — categorized lists like 'Services: PPC, SEO, Social, Analytics'. 4) Call extensions — if you take phone calls, add your number. 5) Image extensions — product or brand images to increase visual appeal. 6) Lead form extensions — capture leads directly from the ad (especially for mobile). Extensions improve CTR by 10-15% on average and they're free — there's no additional cost per click on extensions."},

        # === Remarketing ===
        {"user": "How do I set up effective remarketing campaigns?",
         "assistant": "Segment your audiences by intent level: Tier 1 (highest intent) — cart abandoners, pricing page visitors, free trial signups who didn't convert. Show them specific offer ads. Tier 2 (medium intent) — product/service page visitors, 2+ page visits. Show them testimonials and social proof. Tier 3 (low intent) — all site visitors excluding converters. Show them brand awareness content. Set frequency caps (3-5 impressions per user per day) to avoid ad fatigue. Exclude converters from all remarketing. Use 30-day membership for Tier 1, 90 days for Tier 2-3. Create separate campaigns for each tier with appropriate bids — Tier 1 deserves highest bids."},

        # === Meta Specific ===
        {"user": "How do I set up Meta Ads conversion tracking?",
         "assistant": "Three components: 1) Meta Pixel — install the base pixel code on all pages via your website header or Google Tag Manager. 2) Standard events — add event code for key actions: PageView (all pages), ViewContent (product pages), AddToCart, InitiateCheckout, Purchase (with value), Lead (form submissions). 3) Conversions API (CAPI) — set up server-side event tracking to supplement the Pixel. This is critical post-iOS 14.5 for accurate tracking. Use Meta's Event Manager to verify events are firing. Configure Aggregated Event Measurement by ranking your 8 priority events per domain. Purchase and Lead should be highest priority."},

        {"user": "What's the best Meta Ads audience strategy?",
         "assistant": "Layer audiences in a funnel: Prospecting — use Advantage+ Audience (broad) or Interest-based targeting for new user acquisition. Let Meta's algorithm find converters. Engagement — retarget video viewers (50%+ watched), page engagers, and Instagram interactions. Website retargeting — Pixel-based custom audiences: all visitors (180 days), product viewers (30 days), cart abandoners (14 days). Customer audiences — upload email lists for lookalike creation (1-3% is best for quality). Use exclusions at each level: exclude website visitors from prospecting, exclude purchasers from retargeting."},

        # === Seasonal Strategy ===
        {"user": "How should I adjust Google Ads for Black Friday / holiday season?",
         "assistant": "Start planning 6-8 weeks before: 1) Increase budgets 50-200% based on historical holiday demand. CPCs will rise 20-50% in Q4. 2) Create holiday-specific ad copy with urgency (limited time, ends Sunday, X% off). 3) Build remarketing lists NOW — anyone who visits in October is a warm prospect for November. 4) Launch holiday Shopping campaigns and PMax asset groups with holiday creative. 5) Set up countdown customizers in RSAs for sale end dates. 6) Adjust bid strategy targets — accept higher CPA during peak if CLV justifies it. 7) Pre-schedule ad copy changes for key dates. 8) Monitor hourly during peak days and be ready to increase budgets if campaigns max out."},

        # === Advanced ===
        {"user": "How do I use Google Ads scripts for automation?",
         "assistant": "Google Ads Scripts use JavaScript to automate tasks. Top scripts to implement: 1) Broken URL checker — scans all ad URLs and pauses ads pointing to 404 pages. 2) Quality Score tracker — records daily QS for all keywords in a Google Sheet for trend analysis. 3) Budget pacer — alerts when campaigns are on track to overspend or underspend. 4) Search term miner — automatically adds high-converting search terms as keywords and flags irrelevant terms. 5) Anomaly detector — alerts when metrics deviate significantly from historical norms. Scripts run on schedules (hourly to monthly). Start with the free scripts at developers.google.com/google-ads/scripts or from community repositories."},

        {"user": "Explain the Google Ads API and when to use it vs the UI.",
         "assistant": "The Google Ads API (currently v23) enables programmatic account management. Use the API when: managing 10+ accounts at scale, building custom reporting dashboards, implementing automated bid management, integrating with CRM or inventory systems, or building tools for clients. The API uses GAQL for data retrieval and mutate operations for changes. Authentication requires a developer token, OAuth 2.0 client credentials, and customer ID. Rate limits: 15,000 operations per day for basic access. Python client library: google-ads >= 25.1.0. For single-account management, the Google Ads UI is faster. For anything repetitive across multiple accounts, the API saves enormous time."},

        # === Cross-Platform ===
        {"user": "How do I report performance across Google, Meta, and Microsoft in one view?",
         "assistant": "Create a unified reporting framework: 1) Normalize metrics — Google's CPA = Meta's Cost Per Result = Microsoft's CPA. Google's ROAS = Meta's Purchase ROAS. Map each platform's terms to your standard KPIs. 2) Use a common date range and attribution window. 3) Build a dashboard (Looker Studio, Tableau, or Google Sheets) pulling from each platform's API. Key columns: Platform, Campaign, Spend, Impressions, Clicks, Conversions, CPA, ROAS. 4) Include a blended row showing total across all platforms. 5) Watch for double-counting — a user who clicks Google then Meta before converting might be counted by both platforms. Use UTM parameters and your analytics platform as the source of truth for deduplication."},

        # === Specific Scenarios ===
        {"user": "My client wants to launch in a new market with no historical data. What's the strategy?",
         "assistant": "New market launch strategy: Phase 1 (weeks 1-2, data gathering): start with Maximize Clicks bidding, broad research keywords, and geo-targeting the new market. Budget enough for 100+ clicks per day to gather data fast. Phase 2 (weeks 3-4, refinement): analyze search terms, add negatives, identify converting keywords, switch to phrase/exact match for winners. Phase 3 (weeks 5-8, optimization): once you have 30+ conversions, switch to Maximize Conversions. Begin adding Target CPA after 50+ conversions. Throughout: set up proper conversion tracking before launching, use ad scheduling to match local business hours, and run competitor research via Auction Insights. Expect higher CPA in the first 60 days."},

        {"user": "What is cost_micros in the Google Ads API?",
         "assistant": "cost_micros is Google Ads API's currency format where values are expressed in millionths of the account's currency unit. Divide by 1,000,000 to get the actual amount. Examples: cost_micros = 45000000 means $45.00. cost_micros = 1500000 means $1.50. cost_micros = 100000000 means $100.00. This applies to cost_micros, average_cpc, average_cpm, and all other monetary fields in GAQL responses. When building reports, always divide by 1e6 and format as currency."},
    ]


    pairs.extend(PLATFORM_SFT_PAIRS)
    pairs.extend(BLOG_SFT_PAIRS)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(80):
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"✅ SFT: {path} ({len(pairs)} unique pairs × 100 = {len(pairs) * 100} samples)")


def create_dpo_data():
    """Generate DPO preference pairs — good vs bad advertising advice."""
    path = "./dataset/dpo_ads.jsonl"##===================================
    os.makedirs("./dataset", exist_ok=True)
    pairs = [
        {"prompt": "How should I set up keyword match types?",
         "chosen": "Start with exact match for your highest-intent keywords to control spend and gather clean data. After 2 weeks of performance data, expand top performers to phrase match. Only use broad match with Smart Bidding enabled (Target CPA or ROAS) and robust negative keyword lists. This progression — exact to phrase to broad — ensures you scale spend on proven keywords.",
         "rejected": "Use broad match on everything to maximize traffic volume. More impressions means more clicks means more conversions. Google's AI will figure out what's relevant."},

        {"prompt": "My CPA is too high. What should I do?",
         "chosen": "Diagnose systematically: 1) Check Search Terms Report for irrelevant queries consuming budget — add negatives. 2) Pause keywords with CPA more than 2x your target that haven't improved in 30 days. 3) Audit landing page conversion rate — even small improvements compound. 4) Ensure ad copy matches keyword intent. 5) If using manual CPC, consider switching to Target CPA bidding. Work on the highest-waste areas first.",
         "rejected": "Just increase your budget. More spend means the algorithm has more data to optimize. Google's AI will figure it out eventually if you give it enough money."},

        {"prompt": "Should I run Performance Max campaigns?",
         "chosen": "PMax works best when you have: at least 50 conversions per month, quality creative assets (text, images, video), properly configured conversion tracking, and defined audience signals. Start alongside existing Search campaigns, don't replace them. Use brand exclusions to prevent PMax from cannibalizing your brand traffic. Monitor Search impression share before and after launch.",
         "rejected": "Switch everything to Performance Max immediately. Google recommends it for all advertisers and their AI is always better than manual campaign management. Delete your Search campaigns."},

        {"prompt": "How much should I spend on Google Ads?",
         "chosen": "Work backwards from your goals: if target CPA is $50 and you want 100 leads/month, you need $5,000/month budget. Start at 60-70% of that to test, then scale what works. Each campaign needs minimum $30-50/day to gather meaningful data. Never set a budget you can't sustain for 3 months — Smart Bidding needs consistent data to optimize.",
         "rejected": "Start with $10/day and see what happens. You can always increase later if it works. Keep the budget as low as possible to minimize risk."},

        {"prompt": "Should I target competitor brand names?",
         "chosen": "Competitor targeting can work but requires the right approach: write ads that compare/differentiate rather than mislead, create a dedicated campaign with separate budget, expect higher CPCs and lower Quality Scores (you won't match competitor's relevance), use specific landing pages addressing why you're better. Don't use competitor trademarks in ad text — that violates policies. Monitor ROAS carefully since these campaigns often have lower conversion rates.",
         "rejected": "Always target every competitor name possible and use their brand names in your ad headlines to steal their traffic. This is the fastest way to grow."},

        {"prompt": "How should I handle a sudden spike in CPC?",
         "chosen": "Investigate before reacting: 1) Check Auction Insights for new competitors entering the market. 2) Review Quality Scores for any drops. 3) Look at your Search Impression Share — if budget lost IS increased, your budget may be too low for increased competition. 4) Check if seasonality or industry events are driving more auction demand. If CPCs rose but CVR and CPA stayed stable, the increase might be acceptable. Only reduce bids if efficiency metrics actually worsened.",
         "rejected": "Immediately double your bids to maintain position. You can't afford to lose impression share. Outbid the competition at any cost."},

        {"prompt": "What should I do with keywords that have a Quality Score of 3?",
         "chosen": "QS 3 keywords pay a significant premium per click. Diagnose each component: if Expected CTR is Below Average, improve ad headlines and test more compelling CTAs. If Ad Relevance is Below Average, create a tighter ad group with better keyword-ad alignment. If Landing Page Experience is Below Average, improve page speed, relevance, and mobile experience. Give changes 2-4 weeks to affect QS. If QS doesn't improve after optimization, pause the keyword and find alternative terms.",
         "rejected": "Delete all keywords with Quality Score below 5 immediately. They're dragging down your account performance and there's no way to fix them."},

        {"prompt": "My client wants results in the first week. What do I tell them?",
         "chosen": "Set realistic expectations: Week 1 is data collection, not optimization. Smart Bidding needs 7-14 days for its learning period. Expect higher CPA initially as the algorithm learns. Share a timeline: weeks 1-2 for learning, weeks 3-4 for initial optimization, weeks 5-8 for meaningful results. Show early directional data (search terms, impression share) to demonstrate progress while conversion data accumulates.",
         "rejected": "Promise immediate results to keep the client happy. Set extremely aggressive CPA targets from day one and run broad match keywords to get volume fast."},
    ]


    pairs.extend(PLATFORM_DPO_PAIRS)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(50):
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"✅ DPO: {path} ({len(pairs)} unique pairs × 50 = {len(pairs) * 50} samples)")





def main():
    parser = argparse.ArgumentParser(description="MiniAgent Data & Tokenizer Setup")
    parser.add_argument("--all", action="store_true", help="Download tokenizer + generate all data")
    parser.add_argument("--tokenizer", action="store_true", help="Download minimind tokenizer")
    parser.add_argument("--base-model", action="store_true", help="Download minimind base model for transfer learning")
    # parser.add_argument("--pretrain", action="store_true", help="Generate pretrain corpus")
    # parser.add_argument("--sft", action="store_true", help="Generate SFT instruction pairs")
    # parser.add_argument("--dpo", action="store_true", help="Generate DPO preference pairs")
    parser.add_argument("--pretrain", default=True, help="Generate pretrain corpus")
    parser.add_argument("--sft", default=True, help="Generate SFT instruction pairs")
    parser.add_argument("--dpo", default=True, help="Generate DPO preference pairs")
    args = parser.parse_args()
    # if args.all:##===================================
    #     args.tokenizer = args.base_model = True
    #     args.pretrain = args.sft = args.dpo = True


    # if args.tokenizer:
    #     download_tokenizer()
    # if args.base_model:
    #     download_base_model()
    if args.pretrain:
        create_pretrain_data()
    if args.sft:
        create_sft_data()
    if args.dpo:
        create_dpo_data()
    if not any([args.all, args.tokenizer, args.base_model, args.pretrain, args.sft, args.dpo]):
        print("MiniAgent Data & Tokenizer Setup")
        print("=" * 50)
        print("Usage:")
        print("  python scripts/data_advertising.py --all          # tokenizer + all data")
        print("  python scripts/data_advertising.py --tokenizer    # minimind tokenizer only")
        print("  python scripts/data_advertising.py --base-model   # minimind base model (transfer learning)")
        print("  python scripts/data_advertising.py --pretrain     # pretrain corpus")
        print("  python scripts/data_advertising.py --sft          # SFT instruction pairs")
        print("  python scripts/data_advertising.py --dpo          # DPO preference pairs")
if __name__ == "__main__":
    main()
# #训练数据
# 所有训练数据都是开源的并且特定于广告领域：
# 数据集	              描述	                                            尺寸	    舞台
# pretrain_ads.jsonl	  广告知识库——Google Ads 文档、PPC 最佳实践、行业内容	〜500MB	预训练
# sft_ads.jsonl	      PPC 任务的指令-响应对	                            〜50MB	微调
# sft_gaql.jsonl	      GAQL 查询对 — 自然语言 → GAQL	                    〜10MB	微调
# dpo_ads.jsonl	      好与坏的广告建议对	                                〜20MB	数据好坏
# distill_ads.jsonl	  从 Claude/GPT 的广告任务中提炼出来	                〜100MB	蒸馏
# # Download all datasets
# python scripts/data_advertising.py --all
# # Or specific stages
# python scripts/data_advertising.py --pretrain
# python scripts/data_advertising.py --sft
# python scripts/data_advertising.py --dpo
##
##
# 选项 4：从头开始训练您自己的模型（2 小时，1 个 GPU）
# git clone https://github.com/itallstartedwithaidea/miniagent.git
# cd miniagent
# pip install -r requirements.txt
# # Pretrain (learns advertising language)
# python trainer/pretrain.py --dim 512 --n_layers 8
#
# # SFT (learns to follow advertising instructions)
# python trainer/sft.py --load_from ./checkpoints/pretrain_512.pth
#
# # LoRA fine-tune on YOUR account data (optional)
# python trainer/lora.py --load_from ./checkpoints/sft_512.pth --data ./dataset/my_account.jsonl
# # DPO alignment (learns good vs bad ad advice)
# python trainer/dpo.py --load_from ./checkpoints/sft_512.pth


def ________Dataset():pass
from datasets import load_dataset, Features, Sequence, Value
def _create_sample_dataset(path: str):
    """Create a small sample dataset if full data unavailable."""
    print("For real training, run: python scripts/data_advertising.py --all")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    samples = [
        {"text": "Google Ads uses a real-time auction system where Ad Rank determines position, calculated as Max CPC Bid multiplied by Quality Score plus expected extension impact."},
        {"text": "GAQL queries use SELECT fields FROM resource WHERE conditions. Monetary values use cost_micros, divide by 1,000,000 for actual currency."},
        {"text": "CPA equals total cost divided by total conversions. ROAS equals conversion value divided by cost. Break-even ROAS equals one divided by profit margin."},
        {"text": "Meta Ads uses a campaign-ad set-ad hierarchy with Campaign Budget Optimization. Targeting includes Custom Audiences, Lookalike Audiences, and interest-based targeting."},
        {"text": "Smart Bidding needs at least 30 conversions in 30 days. Target CPA works best with 50+ conversions per month."},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(200):
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Created minimal sample: {path} ({len(samples) * 200} samples)")
class PretrainDataset(Dataset):
    """Advertising pretrain dataset — JSONL format."""
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.samples = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if len(text) > 10:
                    self.samples.append(text)
        print(f"Loaded {len(self.samples)} pretrain samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:self.max_length]
        # Pad to max_length
        padding = self.max_length - len(tokens)
        ##
        input_ids = tokens + [0] * padding
        labels = tokens + [-100] * padding  # -100 = ignore in loss
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }




def _create_sample_sft_dataset(path: str):
    """Create minimal sample SFT data if full data unavailable."""
    print("For real training, run: python scripts/data_advertising.py --all")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    samples = [
        {"user": "What is CPA?", "assistant": "CPA (Cost Per Acquisition) = Total Ad Spend / Total Conversions. If you spent $5,000 and got 100 conversions, your CPA is $50."},
        {"user": "Write a GAQL query for top campaigns by spend.", "assistant": "SELECT campaign.name, metrics.cost_micros, metrics.conversions FROM campaign WHERE segments.date DURING LAST_30_DAYS ORDER BY metrics.cost_micros DESC LIMIT 10."},
        {"user": "My CPA is too high. What should I do?", "assistant": "Check Search Terms Report for irrelevant queries, pause keywords with CPA over 2x target, improve landing page conversion rate, and ensure ad copy matches keyword intent."},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(100):
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Created minimal SFT sample: {path}")
CHAT_TEMPLATE = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
DEFAULT_SYSTEM = "You are MiniAgent, an expert advertising AI assistant specializing in Google Ads, Meta Ads, and cross-platform campaign management."
class SFTDataset(Dataset):
    """Advertising SFT dataset — instruction-response pairs."""
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(item)
        print(f"Loaded {len(self.samples)} SFT samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        system = item.get("system", DEFAULT_SYSTEM)
        user = item.get("user", item.get("instruction", ""))
        assistant = item.get("assistant", item.get("output", ""))
        text = CHAT_TEMPLATE.format(system=system, user=user, assistant=assistant)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = tokens[:self.max_length]
        padding = self.max_length - len(tokens)
        input_ids = tokens + [0] * padding
        labels = tokens + [-100] * padding
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }



