#include <iostream>
#include "llvm/Support/CommandLine.h"

namespace cl = llvm::cl;

static cl::opt<std::string> string_opt(cl::Positional,
                                       cl::desc("<input string option>"),
                                       cl::init("-"),
                                       cl::value_desc("string option"));

static cl::opt<int> integer_opt(cl::Positional,
                                       cl::desc("<input integer option>"),
                                       cl::init(0),
                                       cl::value_desc("integer option"));

static cl::opt<bool> bool_opt("bool-val",
                                       cl::desc("<input boolean option>"),
                                       cl::init(false),
                                       cl::value_desc("boolean option"));

enum EnumVal { V0 = 0, V1, V2, };
static cl::opt<enum EnumVal> enum_opt("enum-val",
                                       cl::desc("<select enum value>"),
                                       cl::init(V0),
                                       cl::values(clEnumValN(V0, "v0", "select V0")),
                                       cl::values(clEnumValN(V1, "v1", "select V1")),
                                       cl::values(clEnumValN(V2, "v2", "select V2")));

template <typename T>
void Print(const T& val) {
  std::cout << val << std::endl;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llvm::cl options");
  Print(string_opt);
  Print(integer_opt);
  Print(bool_opt);
  Print(enum_opt);
  return 0;
}



