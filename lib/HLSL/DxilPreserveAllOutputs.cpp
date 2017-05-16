///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilPreserveAllOutputs.cpp                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Ensure we store to all elements in the output signature.                  //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/HLSL/DxilOperations.h"
#include "dxc/HLSL/DxilSignatureElement.h"
#include "dxc/HLSL/DxilModule.h"
#include "dxc/Support/Global.h"
#include "dxc/HLSL/DxilInstructions.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Pass.h"
#include "llvm/IR/IRBuilder.h"
#include <llvm/ADT/DenseSet.h>

using namespace llvm;
using namespace hlsl;

namespace {
class OutputWrite {
public:
  OutputWrite(CallInst *call)
    : m_Call(call)
  {
    assert(DxilInst_StoreOutput(call) || DxilInst_StorePatchConstant(call));
  }

  unsigned GetSignatureID() const {
    Value *id = m_Call->getOperand(SignatureIndex);
    return cast<ConstantInt>(id)->getLimitedValue();
  }

  SignatureElement &GetSignatureElement(DxilModule &DM) const {
    DM.GetOutputSignature().GetElement(GetSignatureID());
  }

private:
  CallInst *m_Call;
  enum OperandIndex {
    SignatureIndex = 1,
    RowIndex = 2,
    ColumnIndex = 3,
    ValueIndex = 4,
  };
};

class OutputElement {
public:
  OutputElement(const DxilSignatureElement &outputElement)
    : m_OutputElement(outputElement)
    , m_Rows(outputElement.GetRows())
    , m_Columns(outputElement.GetCols())
  {
  }
  OutputElement(const OutputElement&) = delete;
  OutputElement& operator=(const OutputElement&) = delete;

  void CreateAlloca(IRBuilder<> &builder) {
    LLVMContext &context = builder.getContext();
    Type *elementType = m_OutputElement.GetCompType().GetLLVMType(context);
    Type *allocaType = ArrayType::get(elementType, NumElements());
    m_Alloca = builder.CreateAlloca(allocaType);
  }

  void StoreTemp(IRBuilder<> &builder, Value *row, Value *col, Value *value) const {
    Value *GEP = CreateGEP(builder, row, col);
    builder.CreateStore(value, GEP);
  }

  void StoreOutput(IRBuilder<> &builder, DxilModule &DM) const {
    for (unsigned row = 0; row < m_Rows; ++row)
      for (unsigned col = 0; col < m_Columns; ++col) {
        StoreOutput(builder, DM, row, col);
      }
  }

  unsigned NumElements() const {
    return m_Rows * m_Columns;
  }

private:
  const DxilSignatureElement &m_OutputElement;
  unsigned m_Rows;
  unsigned m_Columns;
  AllocaInst *m_Alloca;

  Value *CreateGEP(IRBuilder<> &builder, Value *row, Value *col) const {
    assert(m_Alloca);
    Constant *rowStride = ConstantInt::get(row->getType(), m_Columns);
    Value *rowOffset = builder.CreateMul(row, rowStride);
    Value *index     = builder.CreateAdd(rowOffset, col);
    return builder.CreateInBoundsGEP(m_Alloca, {builder.getInt32(0), index});
  }
  
  Value *LoadTemp(IRBuilder<> &builder, Value *row,  Value *col) const {
    Value *GEP = CreateGEP(builder, row, col);
    builder.CreateLoad(GEP);
  }
  
  void StoreOutput(IRBuilder<> &builder, DxilModule &DM, unsigned row, unsigned col) const {
    Value *opcodeV = builder.getInt32(static_cast<unsigned>(GetOutputOpCode()));
    Value *sigID = builder.getInt32(m_OutputElement.GetID());
    Value *rowV = builder.getInt32(row);
    Value *colV = builder.getInt8(col);
    Value *val = LoadTemp(builder, rowV, colV);
    Value *args[] = { opcodeV, sigID, rowV, colV, val };
    Function *Store = GetOutputFunction(DM);
    builder.CreateCall(Store, args);
  }

  DXIL::OpCode GetOutputOpCode() const {
    if (m_OutputElement.IsPatchConstant())
      return DXIL::OpCode::StorePatchConstant;
    else
      return DXIL::OpCode::StoreOutput;
  }

  Function *GetOutputFunction(DxilModule &DM) const {
    hlsl::OP *opInfo = DM.GetOP();
    return opInfo->GetOpFunc(GetOutputOpCode(), m_OutputElement.GetCompType().GetLLVMBaseType(DM.GetCtx()));
  }
    
};

class DxilPreserveAllOutputs : public FunctionPass {
private:

public:
  static char ID; // Pass identification, replacement for typeid
  DxilPreserveAllOutputs() : FunctionPass(ID) {}

  const char *getPassName() const override {
    return "DXIL preserve all outputs";
  }

  bool runOnFunction(Function &F) override;

#if 0
  bool runOnFunction(Function &F) override {
    DxilModule &DM = M.GetOrCreateDxilModule();
    // Skip pass thru entry.
    if (!DM.GetEntryFunction())
      return false;

    hlsl::OP *hlslOP = DM.GetOP();

    ArrayRef<llvm::Function *> storeOutputs = hlslOP->GetOpFuncList(DXIL::OpCode::StoreOutput);
    DenseMap<Value *, Type *> dynamicSigSet;
    for (Function *F : storeOutputs) {
      // Skip overload not used.
      if (!F)
        continue;
      for (User *U : F->users()) {
        CallInst *CI = cast<CallInst>(U);
        DxilInst_StoreOutput store(CI);
        // Save dynamic indeed sigID.
        if (!isa<ConstantInt>(store.get_rowIndex())) {
          Value * sigID = store.get_outputtSigId();
          dynamicSigSet[sigID] = store.get_value()->getType();
        }
      }
    }

    if (dynamicSigSet.empty())
      return false;

    Function *Entry = DM.GetEntryFunction();
    IRBuilder<> Builder(Entry->getEntryBlock().getFirstInsertionPt());

    DxilSignature &outputSig = DM.GetOutputSignature();
    Value *opcode =
        Builder.getInt32(static_cast<unsigned>(DXIL::OpCode::StoreOutput));
    Value *zero = Builder.getInt32(0);

    for (auto sig : dynamicSigSet) {
      Value *sigID = sig.first;
      Type *EltTy = sig.second;
      unsigned ID = cast<ConstantInt>(sigID)->getLimitedValue();
      DxilSignatureElement &sigElt = outputSig.GetElement(ID);
      unsigned row = sigElt.GetRows();
      unsigned col = sigElt.GetCols();
      Type *AT = ArrayType::get(EltTy, row);

      std::vector<Value *> tmpSigElts(col);
      for (unsigned c = 0; c < col; c++) {
        Value *newCol = Builder.CreateAlloca(AT);
        tmpSigElts[c] = newCol;
      }

      Function *F = hlslOP->GetOpFunc(DXIL::OpCode::StoreOutput, EltTy);
      // Change store output to store tmpSigElts.
      ReplaceDynamicOutput(tmpSigElts, sigID, zero, F);
      // Store tmpSigElts to Output before return.
      StoreTmpSigToOutput(tmpSigElts, row, opcode, sigID, F, Entry);
    }

    return true;
  }
#endif
private:
  typedef std::vector<OutputWrite> CallVec;
  typedef std::vector<ReturnInst *> RetVec;
  typedef std::unordered_map<unsigned, OutputElement>  OutputMap;
  CallVec collectOutputStores(Function &F);
  OutputMap generateOutputMap(const CallVec &calls, DxilModule &DM);
  void createTempAllocas(OutputMap &map, IRBuilder<> &builder);
  void insertTempOutputStores(const CallVec &calls, const OutputMap &map, IRBuilder<> &builder);
};

bool DxilPreserveAllOutputs::runOnFunction(Function &F) {
  DxilModule &DM = F.getParent()->GetOrCreateDxilModule();
  
  CallVec outputStores = collectOutputStores(F);
  if (outputStores.empty())
    return false;

  IRBuilder<> builder(DM.GetCtx());
  OutputMap outputMap = generateOutputMap(outputStores, DM);
  createTempAllocas(outputMap, builder);
  insertTempOutputStores(outputStores, outputMap, builder);
  //insertFinalOutputStores(outputStores, outputMap);
  //removeOriginalOutputStores(outputStores);

  return false;
}

DxilPreserveAllOutputs::CallVec DxilPreserveAllOutputs::collectOutputStores(Function &F) {
  CallVec calls;
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    Instruction *inst = &*I;
    DxilInst_StoreOutput storeOutput(inst);
    DxilInst_StorePatchConstant storePatch(inst);

    if (storeOutput || storePatch)
      calls.push_back(cast<CallInst>(inst));
  }
  return calls;
}

DxilPreserveAllOutputs::OutputMap DxilPreserveAllOutputs::generateOutputMap(const CallVec &calls, DxilModule &DM) {
  OutputMap map;
  for (const OutputWrite &output : calls) {
    if (map.count(output.GetSignatureID()))
      continue;

    map.emplace(output.GetSignatureElement(DM));
  }
}

void DxilPreserveAllOutputs::createTempAllocas(OutputMap &map, IRBuilder<> &builder)
{
  for (auto &iter: map) {
    OutputElement &output = iter.second;
    output.CreateAlloca(builder);
  }
}

void DxilPreserveAllOutputs::insertTempOutputStores(const CallVec &calls, const OutputMap &map, IRBuilder<>& builder)
{
  for (const OutputWrite& outputWrite : calls) {
    auto &iter = map.find(outputWrite.GetSignatureID());
    assert(iter != map.end());
    const OutputElement &output = iter->second;

    builder.SetInsertPoint(outputWrite.getStore());
    output.StoreTemp()
  }
}

#if 0
void DxilEliminateOutputDynamicIndexing::ReplaceDynamicOutput(
    ArrayRef<Value *> tmpSigElts, Value *sigID, Value *zero, Function *F) {
  for (auto it = F->user_begin(); it != F->user_end();) {
    CallInst *CI = cast<CallInst>(*(it++));
    DxilInst_StoreOutput store(CI);
    if (sigID == store.get_outputtSigId()) {
      Value *col = store.get_colIndex();
      unsigned c = cast<ConstantInt>(col)->getLimitedValue();
      Value *tmpSigElt = tmpSigElts[c];
      IRBuilder<> Builder(CI);
      Value *r = store.get_rowIndex();
      // Store to tmpSigElt.
      Value *GEP = Builder.CreateInBoundsGEP(tmpSigElt, {zero, r});
      Builder.CreateStore(store.get_value(), GEP);
      // Remove store output.
      CI->eraseFromParent();
    }
  }
}

void DxilEliminateOutputDynamicIndexing::StoreTmpSigToOutput(
    ArrayRef<Value *> tmpSigElts, unsigned row, Value *opcode, Value *sigID,
    Function *StoreOutput, Function *Entry) {
  Value *args[] = {opcode, sigID, /*row*/ nullptr, /*col*/ nullptr,
                   /*val*/ nullptr};
  // Store the tmpSigElts to Output before every return.
  for (auto &BB : Entry->getBasicBlockList()) {
    if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
      IRBuilder<> Builder(RI);
      Value *zero = Builder.getInt32(0);
      for (unsigned c = 0; c<tmpSigElts.size(); c++) {
        Value *col = tmpSigElts[c];
        args[DXIL::OperandIndex::kStoreOutputColOpIdx] = Builder.getInt8(c);
        for (unsigned r = 0; r < row; r++) {
          Value *GEP =
              Builder.CreateInBoundsGEP(col, {zero, Builder.getInt32(r)});
          Value *V = Builder.CreateLoad(GEP);
          args[DXIL::OperandIndex::kStoreOutputRowOpIdx] = Builder.getInt32(r);
          args[DXIL::OperandIndex::kStoreOutputValOpIdx] = V;
          Builder.CreateCall(StoreOutput, args);
        }
      }
    }
  }
}
#endif
}

char DxilPreserveAllOutputs::ID = 0;

FunctionPass *llvm::createDxilPreserveAllOutputsPass() {
  return new DxilPreserveAllOutputs();
}

INITIALIZE_PASS(DxilPreserveAllOutputs,
                "hlsl-dxil-preserve-all-outputs",
                "DXIL preserve all outputs", false, false)
