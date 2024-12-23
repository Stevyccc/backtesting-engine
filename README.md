# Alpha Factor Selection and Backtesting Framework

## Overview
This is a project that I initially developed for a quantitative asset management internship.
The project implements a custom backtesting framework designed for quantitative trading strategies, specifically focused on alpha factor testing and evaluation. The framework provides a streamlined approach to testing pre-calculated trading signals (vector-based backtesting) and generates comprehensive performance metrics.

## Current Features
- Vector-based backtesting engine for pre-calculated signals
- Standardized input format for signal dataframes
- Comprehensive performance metrics calculation
- Industrial research implementation with multiple signal generation and testing (detailed in accompanying PDF)

## Framework Structure
The backtesting engine accepts signal dataframes with specific required column formats, processes the signals, and outputs important performance metrics. This version (v1) focuses on vector-based backtesting, where signals are pre-calculated before being fed into the framework.

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

2. Dynamic Signal Generation
   - Real-time signal generation capabilities
      - Dynamic signal generation based on market conditions
      - Customizable portfolio construction/basket/strategies/trading rules
   - Price-reactive signal generation
   - Stop Gain/Loss functionality
   - Automated hedging capabilities

3. Advanced Financial Calculations
   - Greeks calculations integration
   - Asset pricing models combining Quantlib
   - Enhanced risk metrics
   - Portfolio optimization and risk management combinging Scipy and cvxpy

## Contributing
This project is currently under active development. Contributions, suggestions, and feedback are welcome.


---
*Note: This is version 1 of the backtesting framework, focused on vector-based signal testing. For a detailed analysis of the industrial research conducted using this framework, please refer to the accompanying PDF document.*
