// Unit tests for the worker line protocol framing -- the ONE definition a worker, the vlm body,
// a client and testworker all share (session_support.{hpp,cpp}: buildFramedLine / parseFramedLine /
// isKnownWorkerOp).  These run on any box (no model, no OpenVINO runtime): they link ovgenai_core
// and exercise the framing + the "no LOAD" invariant directly, so they pass even where the
// subprocess / fixture tests skip.  This is the model-free companion to the --protocol-check run in
// testworker (which drives the framing END-TO-END over the real worker).
//
// The wire framing is  <command>:<session>:<body>  -- the session id is the 2nd header slot (so a
// client reads it WITHOUT parsing the JSON body, then drops the body into its own response envelope),
// the body is everything after the 2nd colon (it may itself hold colons, e.g. a full result JSON),
// and there is deliberately NO load command (the model is loaded on the command line before the loop).
#include <gtest/gtest.h>

#include <optional>

#include "session_support.hpp"    // buildFramedLine / parseFramedLine / isKnownWorkerOp (namespace ovgenai)

using ovgenai::buildFramedLine;
using ovgenai::isKnownWorkerOp;
using ovgenai::parseFramedLine;

namespace {

// buildFramedLine is the plain concatenation <command>:<session>:<body>.
TEST(Framed, buildIsCommandColonSessionColonBody) {
  const std::string line = buildFramedLine("GENERATE", "session1",
                                           R"({"req_id":1,"gen_config":{"messages":[]}})");
  EXPECT_EQ(line, std::string("GENERATE:session1:") +
                  R"({"req_id":1,"gen_config":{"messages":[]}})");
}

// build then parse round-trips the three slots (incl. an empty session slot, as a global op has).
TEST(Framed, buildParseRoundTrip) {
  const std::string line = buildFramedLine("GENERATE", "session1",
                                           R"({"req_id":1,"gen_config":{"messages":[]}})");
  std::string command, session, body;
  ASSERT_TRUE(parseFramedLine(line, command, session, body));
  EXPECT_EQ(command, "GENERATE");
  EXPECT_EQ(session, "session1");
  EXPECT_EQ(body, R"({"req_id":1,"gen_config":{"messages":[]}})");
}

// The crux of the refinement: an empty session slot (a global op like the up-front READY) parses to
// an empty session string, and the body (here an empty object) is kept intact -- no loss of the 2nd/3rd
// delimiter when the middle slot is blank.
TEST(Framed, emptySessionSlot) {
  std::string command, session, body;
  ASSERT_TRUE(parseFramedLine("READY::{}", command, session, body));
  EXPECT_EQ(command, "READY");
  EXPECT_EQ(session, "");       // the 2nd slot is empty, not lost
  EXPECT_EQ(body, "{}");        // the 3rd slot survives on both sides of the empty slot
}

// The body may itself contain colons (a full result JSON does); they are part of the body verbatim,
// NOT delimiters -- only the first two ':' split the header.  This is what lets the body be taken
// straight into a client's response envelope unchanged.
TEST(Framed, bodyWithColonsIsVerbatim) {
  std::string command, session, body;
  const std::string line = "ITEM:3:{\"content\":\"a:b\",\"finish_reason\":\"stop\"}";
  ASSERT_TRUE(parseFramedLine(line, command, session, body));
  EXPECT_EQ(command, "ITEM");
  EXPECT_EQ(session, "3");
  EXPECT_EQ(body, "{\"content\":\"a:b\",\"finish_reason\":\"stop\"}");
}

// A line carrying the two ':' (even with an empty body) is a protocol line.
TEST(Framed, twoColonsIsAProtocolLine) {
  std::string command, session, body;
  ASSERT_TRUE(parseFramedLine("UNLOAD::", command, session, body));
  EXPECT_EQ(command, "UNLOAD");
  EXPECT_EQ(session, "");
  EXPECT_TRUE(body.empty());
}

// A line with no, or only one, ':' is not a protocol line -> parse reports false (the worker answers
// a malformed-line ERROR and keeps running; the testworker drops the stray line).
TEST(Framed, notEnoughDelimitersIsNotAProtocolLine) {
  std::string command, session, body;
  EXPECT_FALSE(parseFramedLine("GENERATE only one colon body", command, session, body));
  EXPECT_FALSE(parseFramedLine("a bare word with no colon", command, session, body));
  EXPECT_FALSE(parseFramedLine("", command, session, body));
}

// The "no LOAD" refinement: the four command words the worker acts on are GENERATE / PING / CANCEL /
// UNLOAD, and a LOAD (or any other word) is not one -- so the worker treats a LOAD as an unknown op.
TEST(Framed, noLoadCommandIsHandled) {
  EXPECT_TRUE(isKnownWorkerOp("GENERATE"));
  EXPECT_TRUE(isKnownWorkerOp("PING"));
  EXPECT_TRUE(isKnownWorkerOp("CANCEL"));
  EXPECT_TRUE(isKnownWorkerOp("UNLOAD"));

  EXPECT_FALSE(isKnownWorkerOp("LOAD"));
  EXPECT_FALSE(isKnownWorkerOp("generate"));   // case-sensitive: the wire is upper-case words
  EXPECT_FALSE(isKnownWorkerOp("BOGUS"));
  EXPECT_FALSE(isKnownWorkerOp(""));
}

// A finished turn's TERMINAL is a single <DONE> line -- the user's decision that a client keys ONLY on
// DONE (whether the turn streamed -- <item>* then this <done> -- or ran with --stream 0, this lone
// <done> -- both shapes end in the SAME key).  Its body is the WHOLE turn (req_id for correlation, then
// content / reasoning_content / tool_calls / finish_reason / metrics), with the session id the 2nd header
// slot and NEVER a body member -- so a client reads the session from the header and takes the body
// verbatim into its own response envelope, and never has to parse two different terminal keys.
TEST(Framed, doneTerminalIsFramedCarriesTheWholeTurnAndNoSessionIdInBody) {
  const std::string body =
      R"({"req_id":7,"content":"Hello","reasoning_content":"thinking","finish_reason":"stop","metrics":{"num_generated_tokens":8}})";
  const std::string line = buildFramedLine("DONE", "s7", body);
  std::string command, session, got;
  ASSERT_TRUE(parseFramedLine(line, command, session, got));
  EXPECT_EQ(command, "DONE");
  EXPECT_EQ(session, "s7");                                        // the session id rides the 2nd header slot
  EXPECT_EQ(got, body);                                           // the whole turn's body is taken verbatim after the 2nd colon
  EXPECT_EQ(got.find("\"session_id\""), std::string::npos);       // the crux: no "session_id" in the body (it is the header)
  EXPECT_NE(got.find("\"content\":"), std::string::npos);         // ...and the body DOES carry the full turn,
  EXPECT_NE(got.find("\"reasoning_content\":"), std::string::npos);
  EXPECT_NE(got.find("\"finish_reason\":"), std::string::npos);   // so a client keyed only on <done> gets everything
  EXPECT_NE(got.find("\"num_generated_tokens\":"), std::string::npos);
}

// Mid-turn, a streamed turn's per-chunk deltas are <ITEM> lines, framed the SAME way (session id in the
// 2nd header slot; the delta body verbatim after the 2nd colon -- a content / reasoning_content /
// tool_calls fragment, none of which is a "session_id").  So a client reading a streamed turn demuxes by
// the header session, accumulates the <item> bodies, then closes on the <done> above -- one reader, the
// session id in neither body.
TEST(Framed, perChunkItemFramesLikeTheTerminalDoneAndCarriesNoSessionId) {
  const std::string body = R"({"content":"el ","reasoning_content":"Let me think:"})";
  const std::string line = buildFramedLine("ITEM", "s7", body);
  std::string command, session, got;
  ASSERT_TRUE(parseFramedLine(line, command, session, got));
  EXPECT_EQ(command, "ITEM");
  EXPECT_EQ(session, "s7");
  EXPECT_EQ(got, body);
  EXPECT_EQ(got.find("\"session_id\""), std::string::npos);
}
}  // namespace
