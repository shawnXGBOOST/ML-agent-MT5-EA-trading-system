//+------------------------------------------------------------------+
//| XAUUSD_Agent_EA.mq5                                              |
//| Phase 7: MT5 EA — reads the Python agent's signal bridge file,   |
//| independently validates and enforces risk, places pending limit  |
//| orders with SL/TP and expiry. Per the blueprint's core principle:|
//| "the agent proposes, the EA enforces." Nothing here trusts the   |
//| Python side blindly — every hard risk check is re-verified here. |
//|                                                                    |
//| NOTE: this has not been compiled/tested in MetaEditor (no MT5    |
//| available in the environment this was written in). Compile it,   |
//| and paste back any errors — they'll be fixed from the messages.  |
//+------------------------------------------------------------------+
#property copyright "Phase 7"
#property version   "1.00"
#property strict

//--- Inputs
input string InpSignalFilePath      = "signal.json";     // relative to MQL5/Files/
input int    InpMaxSpreadPoints     = 300;                // reject if current spread exceeds this
input double InpMaxLotSize          = 0.05;                // absolute hard cap regardless of risk calc
input double InpMaxDailyLossPct     = 4.0;                 // matches Phase 0
input double InpMaxDrawdownPct      = 15.0;                // matches Phase 0
input int    InpMaxSignalAgeSeconds = 120;                 // reject signal older than this
input int    InpMagicNumber         = 778001;
input int    InpCooldownSeconds     = 300;                 // don't re-place a near-identical setup within this window
input double InpCooldownPriceBand   = 0.50;                // "near-identical" = entry within this many $ of last order
// --- Stop-lock exit (must match triple_barrier_staged.py settings) ---
input bool   InpEnableStopLock      = true;
input double InpActivationFraction  = 0.4;   // activate at this fraction of entry->TP distance
input double InpLockFraction        = 1.0;   // lock this fraction of the activation distance (<=1.0)
input bool   InpCloseIfLockBreached = true;  // if activation was hit but price already fell back
                                             // past the lock level, close at market (the stop
                                             // would have been hit there under the validated rule)

//--- State (persisted across ticks via globals; EA restart resets these —
//    acceptable since MT5 terminal restart is rare mid-session, and the
//    Python-side sequence_id file survives independently)
long   g_lastProcessedSequenceId = 0;
double g_dayStartEquity          = 0.0;
datetime g_currentDay            = 0;
double g_peakEquity              = 0.0;
bool   g_drawdownHalted          = false;
datetime g_lastOrderTime         = 0;
double g_lastOrderEntryPrice     = 0.0;

datetime g_lastStopLockLog = 0;   // throttles repeated failure logs

//+------------------------------------------------------------------+
int OnInit()
  {
   g_peakEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayStartEquity = g_peakEquity;
   g_currentDay = TimeCurrent();
   EventSetTimer(15); // poll every 15 seconds — plenty fast for a 15-min decision cycle
   Print("XAUUSD_Agent_EA initialized. Watching: ", InpSignalFilePath);
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTick()
  {
   // Risk state is updated on tick so daily/drawdown checks stay current
   // even between the 15s timer polls.
   RollDayIfNeeded();
   UpdateDrawdownState();
   ManagePositionLifecycle(); // checked every tick — activation timing matters,
                               // shouldn't wait for the 15s timer
  }

void OnTimer()
  {
   RollDayIfNeeded();
   UpdateDrawdownState();
   ProcessSignalFile();
  }

//+------------------------------------------------------------------+
//| Day rollover — mirrors the Python agent's logic so both sides    |
//| agree on what "today" means for the daily loss limit.            |
//+------------------------------------------------------------------+
void RollDayIfNeeded()
  {
   MqlDateTime now, last;
   TimeToStruct(TimeCurrent(), now);
   TimeToStruct(g_currentDay, last);
   if(now.day != last.day || now.mon != last.mon || now.year != last.year)
     {
      g_currentDay = TimeCurrent();
      g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
     }
  }

void UpdateDrawdownState()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_peakEquity)
      g_peakEquity = equity;

   double ddPct = (g_peakEquity > 0) ? (g_peakEquity - equity) / g_peakEquity * 100.0 : 0.0;
   if(ddPct >= InpMaxDrawdownPct)
     {
      if(!g_drawdownHalted)
         Print("DRAWDOWN CIRCUIT BREAKER TRIGGERED: ", DoubleToString(ddPct,2),
               "% >= ", DoubleToString(InpMaxDrawdownPct,2), "% — halting all new entries.");
      g_drawdownHalted = true;
     }
  }

double DailyLossPct()
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(g_dayStartEquity <= 0) return 0.0;
   double loss = (g_dayStartEquity - equity) / g_dayStartEquity * 100.0;
   return MathMax(0.0, loss);
  }

//+------------------------------------------------------------------+
//| Minimal JSON field extraction — tailored to our fixed, flat      |
//| schema (no general parser needed since we control the writer).   |
//+------------------------------------------------------------------+
string JsonGetString(string json, string key)
  {
   string pattern = "\"" + key + "\"";
   int pos = StringFind(json, pattern);
   if(pos < 0) return "";
   pos = StringFind(json, ":", pos);
   if(pos < 0) return "";
   int quoteStart = StringFind(json, "\"", pos);
   if(quoteStart < 0) return "";
   int quoteEnd = StringFind(json, "\"", quoteStart + 1);
   if(quoteEnd < 0) return "";
   return StringSubstr(json, quoteStart + 1, quoteEnd - quoteStart - 1);
  }

double JsonGetNumber(string json, string key)
  {
   string pattern = "\"" + key + "\"";
   int pos = StringFind(json, pattern);
   if(pos < 0) return 0.0;
   pos = StringFind(json, ":", pos);
   if(pos < 0) return 0.0;
   pos++;
   // skip whitespace
   while(pos < StringLen(json) && (StringGetCharacter(json, pos) == ' '))
      pos++;
   int end = pos;
   while(end < StringLen(json))
     {
      ushort c = StringGetCharacter(json, end);
      if(c == ',' || c == '\n' || c == '\r' || c == '}')
         break;
      end++;
     }
   string numStr = StringSubstr(json, pos, end - pos);
   StringTrimRight(numStr);
   StringTrimLeft(numStr);
   return StringToDouble(numStr);
  }

long JsonGetInt(string json, string key)
  {
   return (long)JsonGetNumber(json, key);
  }

//+------------------------------------------------------------------+
//| Read + validate + act on the signal file                         |
//+------------------------------------------------------------------+
void ProcessSignalFile()
  {
   if(g_drawdownHalted)
      return; // requires manual review/restart per Phase 0 design — no auto-resume

   if(DailyLossPct() >= InpMaxDailyLossPct)
      return; // daily halt, resumes automatically next day via RollDayIfNeeded

   if(PositionsTotalForSymbolMagic() > 0)
      return; // max concurrent = 1, already in a position — matches Phase 0

   int fh = FileOpen(InpSignalFilePath, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
      return; // no signal file yet, or not readable — not an error, just nothing to do

   string content = "";
   while(!FileIsEnding(fh))
      content += FileReadString(fh) + "\n";
   FileClose(fh);

   if(StringLen(content) < 10)
      return;

   long sequenceId = JsonGetInt(content, "sequence_id");
   if(sequenceId <= g_lastProcessedSequenceId)
      return; // already processed this one, or file is stale/reverted

   string writtenAtStr = JsonGetString(content, "written_at_utc");
   datetime writtenAt = StringToTime(writtenAtStr); // MQL5-native "YYYY.MM.DD HH:MI:SS" format
   int ageSeconds = (int)(TimeGMT() - writtenAt);
   if(ageSeconds > InpMaxSignalAgeSeconds || ageSeconds < -60)
     {
      Print("REJECTED stale/invalid signal seq=", sequenceId, " age=", ageSeconds, "s");
      g_lastProcessedSequenceId = sequenceId; // don't keep re-rejecting the same stale one
      return;
     }

   string symbol = JsonGetString(content, "symbol");
   string direction = JsonGetString(content, "direction");
   double entryPrice = JsonGetNumber(content, "entry_price");
   double slPrice = JsonGetNumber(content, "sl_price");
   double tpPrice = JsonGetNumber(content, "tp_price");
   double riskPct = JsonGetNumber(content, "risk_pct");
   int expiryMinutes = (int)JsonGetNumber(content, "expiry_minutes");
   // Staged-exit fields — optional for backward compatibility with older
   // signal files that don't have them (activationPrice=0 means "no staged
   // exit, behave like a plain fixed TP/SL trade").
   double activationPrice = JsonGetNumber(content, "activation_price");
   double lockPrice = JsonGetNumber(content, "lock_price");

   g_lastProcessedSequenceId = sequenceId;

   if(symbol != _Symbol)
     {
      Print("REJECTED signal for symbol ", symbol, " (this EA runs on ", _Symbol, ")");
      return;
     }

   if(entryPrice <= 0 || slPrice <= 0 || tpPrice <= 0)
     {
      Print("REJECTED signal seq=", sequenceId, " — invalid price levels");
      return;
     }

   // Spread check — independent hard gate, not something the agent evaluated
   long spreadPoints = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   if(spreadPoints > InpMaxSpreadPoints)
     {
      Print("REJECTED signal seq=", sequenceId, " — spread ", spreadPoints,
            " exceeds max ", InpMaxSpreadPoints);
      return;
     }

   // Cooldown — avoid repeatedly placing near-identical setups
   if(TimeCurrent() - g_lastOrderTime < InpCooldownSeconds &&
      MathAbs(entryPrice - g_lastOrderEntryPrice) < InpCooldownPriceBand)
     {
      Print("SKIPPED signal seq=", sequenceId, " — cooldown active for near-identical setup");
      return;
     }

   PlacePendingOrder(direction, entryPrice, slPrice, tpPrice, riskPct, expiryMinutes, sequenceId,
                      activationPrice, lockPrice);
  }

//+------------------------------------------------------------------+
//| Position sizing with broker-specific lot rounding — this is      |
//| where the small-account granularity issue we identified gets     |
//| handled explicitly rather than silently.                         |
//+------------------------------------------------------------------+
double CalculateLotSize(double riskPct, double entryPrice, double slPrice, bool &riskExceeded)
  {
   riskExceeded = false;
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskDollar = equity * (riskPct / 100.0);

   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double slDistance = MathAbs(entryPrice - slPrice);

   if(tickSize <= 0 || slDistance <= 0)
     {
      Print("CalculateLotSize: invalid tickSize/slDistance, aborting sizing");
      return 0.0;
     }

   double lossPerLot = (slDistance / tickSize) * tickValue;
   double rawLot = riskDollar / lossPerLot;

   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   // Round DOWN to the nearest lot step — never round up past intended risk
   double lot = MathFloor(rawLot / lotStep) * lotStep;
   lot = MathMin(lot, InpMaxLotSize);
   lot = MathMin(lot, maxLot);

   if(lot < minLot)
     {
      // Small-account reality check: even the minimum lot may risk more than
      // intended. Flag it rather than silently over-risking.
      double minLotRiskDollar = lossPerLot * minLot;
      double minLotRiskPct = (equity > 0) ? (minLotRiskDollar / equity * 100.0) : 999;
      Print("WARNING: minimum lot (", minLot, ") would risk ", DoubleToString(minLotRiskPct,2),
            "% vs intended ", DoubleToString(riskPct,2), "% — ",
            (minLotRiskPct > riskPct * 1.5 ? "EXCEEDS 1.5x intended, skipping trade" : "using minimum lot anyway"));
      if(minLotRiskPct > riskPct * 1.5)
        {
         riskExceeded = true;
         return 0.0;
        }
      return minLot;
     }

   return lot;
  }

//+------------------------------------------------------------------+
//| Validate the pending order price against LIVE market conditions  |
//| before sending — this is what actually diagnoses retcode 10015   |
//| (Invalid price) instead of just logging a generic failure.       |
//+------------------------------------------------------------------+
bool ValidateOrderPrice(ENUM_ORDER_TYPE orderType, double &entryPrice, string &rejectReason)
  {
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   long stopsLevelPoints = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDistance = stopsLevelPoints * point;

   if(orderType == ORDER_TYPE_BUY_LIMIT)
     {
      if(entryPrice >= ask)
        {
         rejectReason = StringFormat("BUY LIMIT price %.2f is at/above current ask %.2f — "
                                      "market moved past this level since the signal was computed",
                                      entryPrice, ask);
         return false;
        }
      if((ask - entryPrice) < minDistance)
        {
         rejectReason = StringFormat("BUY LIMIT price %.2f is only %.2f from ask %.2f — "
                                      "below broker minimum distance %.2f",
                                      entryPrice, ask - entryPrice, ask, minDistance);
         return false;
        }
     }
   else // SELL_LIMIT
     {
      if(entryPrice <= bid)
        {
         rejectReason = StringFormat("SELL LIMIT price %.2f is at/below current bid %.2f — "
                                      "market moved past this level since the signal was computed",
                                      entryPrice, bid);
         return false;
        }
      if((entryPrice - bid) < minDistance)
        {
         rejectReason = StringFormat("SELL LIMIT price %.2f is only %.2f from bid %.2f — "
                                      "below broker minimum distance %.2f",
                                      entryPrice, entryPrice - bid, bid, minDistance);
         return false;
        }
     }
   return true;
  }

//+------------------------------------------------------------------+
void PlacePendingOrder(string direction, double entryPrice, double slPrice, double tpPrice,
                        double riskPct, int expiryMinutes, long sequenceId,
                        double activationPrice, double lockPrice)
  {
   bool riskExceeded = false;
   double lot = CalculateLotSize(riskPct, entryPrice, slPrice, riskExceeded);
   if(riskExceeded || lot <= 0)
     {
      Print("SKIPPED signal seq=", sequenceId, " — sizing rejected (see warning above)");
      return;
     }

   ENUM_ORDER_TYPE orderType = (direction == "long") ? ORDER_TYPE_BUY_LIMIT : ORDER_TYPE_SELL_LIMIT;

   string rejectReason = "";
   if(!ValidateOrderPrice(orderType, entryPrice, rejectReason))
     {
      Print("SKIPPED signal seq=", sequenceId, " — ", rejectReason);
      return;
     }

   MqlTradeRequest request;
   MqlTradeResult  result;
   ZeroMemory(request);
   ZeroMemory(result);

   request.action       = TRADE_ACTION_PENDING;
   request.symbol        = _Symbol;
   request.volume        = lot;
   request.type          = orderType;
   request.price         = NormalizeDouble(entryPrice, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
   request.sl             = NormalizeDouble(slPrice, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
   request.tp             = NormalizeDouble(tpPrice, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
   request.type_time      = ORDER_TIME_SPECIFIED;
   request.expiration     = TimeCurrent() + expiryMinutes * 60;
   request.type_filling   = ORDER_FILLING_RETURN;
   request.magic          = InpMagicNumber;
   request.comment        = "seq" + IntegerToString(sequenceId);

   bool sent = OrderSend(request, result);
   if(!sent || result.retcode != TRADE_RETCODE_DONE)
     {
      Print("ORDER FAILED seq=", sequenceId, " retcode=", result.retcode, " comment=", result.comment);
      return;
     }

   g_lastOrderTime = TimeCurrent();
   g_lastOrderEntryPrice = entryPrice;

   Print("ORDER PLACED seq=", sequenceId, " ", direction, " lot=", lot,
         " entry=", entryPrice, " sl=", slPrice, " tp=", tpPrice,
         " expiry=", expiryMinutes, "min",
         (activationPrice > 0 ? " activation=" + DoubleToString(activationPrice, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) : ""),
         (lockPrice > 0 ? " lock=" + DoubleToString(lockPrice, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)) : ""));
  }

//+------------------------------------------------------------------+
//| Stop-lock management — STATELESS, checked every tick.            |
//| For every open position of this symbol + magic number, recompute |
//| activation/lock from the position's own entry price and TP, check|
//| the best price reached since it opened, and move the SL once.    |
//| Nothing is kept in memory, so EA reloads, MT5 restarts, or       |
//| missing activation fields in the signal file can't break it.     |
//+------------------------------------------------------------------+
void ManagePositionLifecycle()
  {
   if(!InpEnableStopLock) return;
   if(InpLockFraction > 1.0 || InpLockFraction <= 0 || InpActivationFraction <= 0) return;

   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double point  = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double bid    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double spread = ask - bid;
   long   stopsLvl  = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   long   freezeLvl = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   // Minimum distance a new SL must keep from current price: broker level,
   // but never less than one spread (a zero-buffer stop gets rejected or
   // clipped by noise immediately).
   double minDist = MathMax((double)MathMax(stopsLvl, freezeLvl) * point, spread);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      long     type     = PositionGetInteger(POSITION_TYPE);
      double   entry    = PositionGetDouble(POSITION_PRICE_OPEN);
      double   curSL    = PositionGetDouble(POSITION_SL);
      double   tp       = PositionGetDouble(POSITION_TP);
      datetime openTime = (datetime)PositionGetInteger(POSITION_TIME);
      if(tp <= 0) continue; // no TP -> can't derive the geometry

      double tpDist   = MathAbs(tp - entry);
      double actDist  = InpActivationFraction * tpDist;
      double lockDist = actDist * InpLockFraction;
      bool   isLong   = (type == POSITION_TYPE_BUY);

      double activation = isLong ? entry + actDist  : entry - actDist;
      double lockPrice  = isLong ? entry + lockDist : entry - lockDist;

      // Already locked? (SL at or beyond break-even-plus side of entry)
      if(isLong  && curSL > 0 && curSL >= entry + 0.5 * point) continue;
      if(!isLong && curSL > 0 && curSL <= entry - 0.5 * point) continue;

      // Best price reached since the position opened (M1 history + now).
      double best;
      if(isLong)
        {
         double highs[];
         best = bid;
         int n = CopyHigh(_Symbol, PERIOD_M1, openTime, TimeCurrent(), highs);
         for(int k = 0; k < n; k++) best = MathMax(best, highs[k]);
         if(best < activation) continue;
        }
      else
        {
         double lows[];
         best = ask;
         int n = CopyLow(_Symbol, PERIOD_M1, openTime, TimeCurrent(), lows);
         // chart lows are bid prices; a short exits at ask, so add spread
         for(int k = 0; k < n; k++) best = MathMin(best, lows[k] + spread);
         if(best > activation) continue;
        }

      // Activation was reached. Work out the SL we can actually place now.
      double exitPx = isLong ? bid : ask;
      bool   breached = isLong ? (exitPx <= lockPrice) : (exitPx >= lockPrice);
      double newSL;

      if(breached)
        {
         // Price already fell back past the lock level: under the rule the
         // stop would have fired there. Close now rather than give it back.
         if(InpCloseIfLockBreached)
           {
            MqlTradeRequest cr; MqlTradeResult cres;
            ZeroMemory(cr); ZeroMemory(cres);
            cr.action    = TRADE_ACTION_DEAL;
            cr.position  = ticket;
            cr.symbol    = _Symbol;
            cr.volume    = PositionGetDouble(POSITION_VOLUME);
            cr.type      = isLong ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
            cr.price     = exitPx;
            cr.deviation = 30;
            cr.magic     = InpMagicNumber;
            long fm = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
            if((fm & SYMBOL_FILLING_FOK) == SYMBOL_FILLING_FOK)      cr.type_filling = ORDER_FILLING_FOK;
            else if((fm & SYMBOL_FILLING_IOC) == SYMBOL_FILLING_IOC) cr.type_filling = ORDER_FILLING_IOC;
            else                                                     cr.type_filling = ORDER_FILLING_RETURN;
            if(OrderSend(cr, cres) && (cres.retcode == TRADE_RETCODE_DONE || cres.retcode == TRADE_RETCODE_PLACED))
               Print("STOP-LOCK: activation was hit but price fell back past lock ",
                     DoubleToString(lockPrice, digits), " — closed ticket=", ticket, " at ", exitPx);
            else if(TimeCurrent() - g_lastStopLockLog > 30)
              {
               Print("STOP-LOCK close FAILED ticket=", ticket, " retcode=", cres.retcode, " ", cres.comment);
               g_lastStopLockLog = TimeCurrent();
              }
           }
         continue;
        }

      // Place SL at the lock level, pulled back if it's too close to price.
      if(isLong) newSL = MathMin(lockPrice, exitPx - minDist);
      else       newSL = MathMax(lockPrice, exitPx + minDist);
      newSL = NormalizeDouble(newSL, digits);

      // Only worth moving if it still locks in profit.
      if(isLong  && newSL <= entry) continue;
      if(!isLong && newSL >= entry) continue;

      MqlTradeRequest req; MqlTradeResult res;
      ZeroMemory(req); ZeroMemory(res);
      req.action   = TRADE_ACTION_SLTP;
      req.position = ticket;
      req.symbol   = _Symbol;
      req.sl       = newSL;
      req.tp       = tp; // resend TP unchanged
      req.magic    = InpMagicNumber;

      if(OrderSend(req, res) && res.retcode == TRADE_RETCODE_DONE)
         Print("STOP MOVED ticket=", ticket, " entry=", DoubleToString(entry, digits),
               " activation=", DoubleToString(activation, digits),
               " new SL=", DoubleToString(newSL, digits),
               (newSL != NormalizeDouble(lockPrice, digits) ? " (pulled back from lock for min distance)" : ""));
      else if(TimeCurrent() - g_lastStopLockLog > 30)
        {
         Print("STOP MOVE FAILED ticket=", ticket, " wanted SL=", DoubleToString(newSL, digits),
               " retcode=", res.retcode, " ", res.comment, " — retrying on next ticks");
         g_lastStopLockLog = TimeCurrent();
        }
     }
  }

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Counts positions/pending orders for THIS SYMBOL and magic number |
//| only. Each instrument gets its own independent concurrency slot  |
//| — attach this EA to XAUUSD and NAS100 charts with the same       |
//| magic number, and each trades independently without blocking    |
//| the other (reverted from an earlier shared-concurrency design).  |
//+------------------------------------------------------------------+
int PositionsTotalForSymbolMagic()
  {
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket <= 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         count++;
     }
   for(int i = 0; i < OrdersTotal(); i++)
     {
      ulong ticket = OrderGetTicket(i);
      if(ticket <= 0) continue;
      if(OrderGetString(ORDER_SYMBOL) == _Symbol &&
         OrderGetInteger(ORDER_MAGIC) == InpMagicNumber)
         count++;
     }
   return count;
  }
//+------------------------------------------------------------------+
