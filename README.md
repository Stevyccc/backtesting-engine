# Alpha Factor Selection and Backtesting Framework

## Overview
This is a project that I initially developed during a quantitative asset management internship where I worked on a backtesting framework for evaluating alpha factors. This has helped me greatly in understanding the in-house sophisticated industry-level backtesting object oriented framework during my role as a quantitative analyst at SigTech (spin-off of hedge fund Brevan Howard), which is a much more robust framework that covers from data cleaning, asset pricing, trading cost, dynamic hedging, portfolio optimization all the way to sensitivity testing.
The project implements a custom backtesting framework designed for quantitative trading strategies, specifically focused on alpha factor testing and evaluation. The framework provides a streamlined approach to testing pre-calculated trading signals (vector-based backtesting) and generates comprehensive performance metrics.

## Current Features
- Vector-based backtesting engine for pre-calculated signals
- Standardized input format for signal dataframes
- Comprehensive performance metrics calculation
- Industrial research implementation with multiple signal generation and testing (detailed in accompanying PDF)

## Framework Structure
The backtesting engine accepts signal dataframes with specific required column formats, processes the signals, and outputs important performance metrics. This version (v1) focuses on vector-based backtesting, where signals are pre-calculated before being fed into the framework.

## Detailed Functionality and Parameters

### Input Requirements
- **Signal DataFrame Format**:
  - Trading date column ('TrdDt')
  - Signal values for each sector/asset
  - Price data ('close')
  - Volume data for liquidity analysis
  - Sector classification data

### Key Features and Parameters
1. **Signal Processing and Ranking**:
   - Percentile-based cross-sectional ranking (`pct_ranking` function)
   - Customizable number of ranking groups (quantiles)
   - Configurable signal delay implementation
   - Flexible signal scoring and normalization

2. **Trading Schedule Management**:
   - Two rebalancing algorithms:
     - Natural Day Based (`rebalance_dates`)
     - Trading Day Based (`rebalance_dates_v2`)
   - Customizable rebalancing frequency
   - Support for custom rebalancing dates
   - Rolling window analysis for strategy timing sensitivity

3. **Portfolio Analysis**:
   - Group-based portfolio construction
   - Maximum drawdown calculation
   - Trading day difference calculations
   - Sector-based analysis and rotation
   - Performance tracking across different ranking groups

4. **Advanced Features**:
   - Rolling window sensitivity analysis
   - Forward/backward testing capabilities
   - Sector rotation strategies
   - Trading delay implementation
   - Flexible date handling and conversion

### Performance Metrics
- Returns across different ranking groups
- Maximum drawdown analysis
- Trading frequency statistics
- Sector exposure analysis
- Signal decay analysis through rolling windows

### Unique Capabilities
- Support for both natural and trading day-based rebalancing (to accomodate for differet market holidays, ideally a trading schedule should be provided)
- Built-in sector rotation functionality
- Flexible signal delay implementation for realistic backtesting
- Rolling window analysis for strategy timing optimization
- Cross-sectional ranking with customizable quantiles

## Research Implementation
As part of this project, an industrial research study was conducted using the backtesting framework to:
- Develop understanding of the signal generation and evaluation process
- Generate and test multiple trading signals
- Evaluate signal performance
- Analyze strategy effectiveness
(For detailed findings and methodology, please refer to the included PDF document)

## Future Development Plans
### Version 2 Roadmap
1. Object-Oriented Architecture
   - Restructure the framework using OOP principles
   - Simplify code organization and run-time efficiency by vectorization
   - Enhance code maintainability by applying SOLID principles
   - Improve code modularity and reusability

2. Systematic Trading Calendar
   - Implement a systematic trading calendar timeline
   - Integrate market holidays and trading sessions

3. Dynamic Signal Generation
   - Real-time signal generation capabilities
      - Dynamic signal generation based on market conditions
      - Customizable portfolio construction/basket/strategies/trading rules
   - Price-reactive signal generation
   - Stop Gain/Loss functionality
   - Automated hedging capabilities

4. Advanced Financial Calculations
   - Greeks calculations integration
   - Asset pricing models combining Quantlib
   - Enhanced risk metrics
   - Portfolio optimization and risk management combinging Scipy and cvxpy

## Contributing
This project is currently under active development. Contributions, suggestions, and feedback are welcome.


---
*Note: This is version 1 of the backtesting framework, focused on vector-based signal testing. The code actually includes the full framework but only a few factors data that are mentioned in the report.For a detailed analysis of the industrial research conducted using this framework, please refer to the accompanying PDF document.*
