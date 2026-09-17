import { describe, expect, it, vi } from 'vitest'

import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import {
  applyVoiceRecordResponse,
  dismissSensitivePrompt,
  handleIdleHotkeyExit,
  resolveCtrlCComposerAction,
  shouldAllowIdleHotkeyExit,
  shouldDetachEditedHistoryInput,
  shouldFallThroughForScroll
} from '../app/useInputHandlers.js'

const baseKey = {
  downArrow: false,
  pageDown: false,
  pageUp: false,
  shift: false,
  upArrow: false,
  wheelDown: false,
  wheelUp: false
}

describe('shouldFallThroughForScroll — keep transcript scrolling alive during prompt overlays', () => {
  it('falls through for wheel scrolls', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, wheelUp: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, wheelDown: true })).toBe(true)
  })

  it('falls through for PageUp / PageDown', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, pageUp: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, pageDown: true })).toBe(true)
  })

  it('falls through for Shift+ArrowUp / Shift+ArrowDown', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true, upArrow: true })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true, downArrow: true })).toBe(true)
  })

  it('does NOT fall through for plain arrows — those drive in-prompt selection', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, upArrow: true })).toBe(false)
    expect(shouldFallThroughForScroll({ ...baseKey, downArrow: true })).toBe(false)
  })

  it('does NOT fall through for plain Shift — without an arrow it is a no-op', () => {
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true })).toBe(false)
  })

  it('does NOT fall through for unrelated state (no scroll keys held)', () => {
    expect(shouldFallThroughForScroll(baseKey)).toBe(false)
  })
})

describe('shouldAllowIdleHotkeyExit', () => {
  it('keeps idle exit hotkeys enabled in normal terminals', () => {
    expect(shouldAllowIdleHotkeyExit(false)).toBe(true)
  })

  it('disables idle exit hotkeys in dashboard chat', () => {
    expect(shouldAllowIdleHotkeyExit(true)).toBe(false)
  })
})

describe('shouldDetachEditedHistoryInput', () => {
  const history = ['older message', 'line one\nline two']

  it('detaches a recalled entry as soon as the user edits it', () => {
    expect(shouldDetachEditedHistoryInput(1, history, 'line one edited\nline two')).toBe(true)
  })

  it('keeps unchanged recalled entries in history navigation', () => {
    expect(shouldDetachEditedHistoryInput(1, history, 'line one\nline two')).toBe(false)
  })

  it('does not detach an ordinary current draft', () => {
    expect(shouldDetachEditedHistoryInput(null, history, 'new draft')).toBe(false)
  })
})

describe('resolveCtrlCComposerAction — draft wins over interrupt', () => {
  it('clears a non-empty composer even while the agent is streaming', () => {
    expect(resolveCtrlCComposerAction({ busy: true, hasDraft: true, hasSession: true })).toBe('clear')
  })

  it('interrupts a running turn when the composer is empty', () => {
    expect(resolveCtrlCComposerAction({ busy: true, hasDraft: false, hasSession: true })).toBe('interrupt')
  })

  it('clears an idle composer instead of exiting', () => {
    expect(resolveCtrlCComposerAction({ busy: false, hasDraft: true, hasSession: true })).toBe('clear')
  })

  it('exits when idle with an empty composer', () => {
    expect(resolveCtrlCComposerAction({ busy: false, hasDraft: false, hasSession: true })).toBe('exit')
  })

  it('does not interrupt a busy session that has no sid yet', () => {
    expect(resolveCtrlCComposerAction({ busy: true, hasDraft: false, hasSession: false })).toBe('exit')
  })
})

describe('handleIdleHotkeyExit', () => {
  it('exits in normal terminals', () => {
    const actions = { die: vi.fn(), sys: vi.fn() }

    handleIdleHotkeyExit(actions, false)

    expect(actions.die).toHaveBeenCalledTimes(1)
    expect(actions.sys).not.toHaveBeenCalled()
  })

  it('asks the dashboard for a fresh chat instead of leaving a ghost session', () => {
    const actions = { die: vi.fn(), sys: vi.fn() }
    const requestDashboardNewSession = vi.fn()

    handleIdleHotkeyExit(actions, true, requestDashboardNewSession)

    expect(actions.die).not.toHaveBeenCalled()
    expect(requestDashboardNewSession).toHaveBeenCalledTimes(1)
    expect(actions.sys).toHaveBeenCalledWith('starting a fresh dashboard chat...')
  })
})

describe('applyVoiceRecordResponse', () => {
  it('reverts optimistic REC state when the gateway reports voice busy', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()
    const sys = vi.fn()

    applyVoiceRecordResponse({ status: 'busy' }, true, { setProcessing, setRecording }, sys)

    expect(setRecording).toHaveBeenCalledWith(false)
    expect(setProcessing).toHaveBeenCalledWith(true)
    expect(sys).toHaveBeenCalledWith('voice: still transcribing; try again shortly')
  })

  it('keeps optimistic REC state for successful recording starts', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()

    applyVoiceRecordResponse({ status: 'recording' }, true, { setProcessing, setRecording }, vi.fn())

    expect(setRecording).not.toHaveBeenCalled()
    expect(setProcessing).not.toHaveBeenCalled()
  })

  it('reverts optimistic REC state when the gateway returns null', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()

    applyVoiceRecordResponse(null, true, { setProcessing, setRecording }, vi.fn())

    expect(setRecording).toHaveBeenCalledWith(false)
    expect(setProcessing).toHaveBeenCalledWith(false)
  })
})

describe('dismissSensitivePrompt', () => {
  it('ignores a stale verification-code cancellation after the prompt is replaced', async () => {
    resetOverlayState()
    patchOverlayState({ vaultCode: { requestId: 'code-a', site: 'old.example' } })
    const staleOverlay = getOverlayState()

    patchOverlayState({ vaultCode: { requestId: 'code-b', site: 'new.example' } })
    const rpc = vi.fn().mockResolvedValue(null)

    await dismissSensitivePrompt(staleOverlay, rpc, vi.fn())

    expect(rpc).not.toHaveBeenCalled()
    expect(getOverlayState().vaultCode?.requestId).toBe('code-b')
  })

  it('ignores a stale login-save cancellation after the prompt is replaced', async () => {
    resetOverlayState()
    patchOverlayState({
      vaultSaveLogin: {
        identifier: 'old-user',
        origin: 'https://old.example',
        requestId: 'save-a',
        site: 'old.example',
        step: 'password'
      }
    })
    const staleOverlay = getOverlayState()

    patchOverlayState({
      vaultSaveLogin: {
        identifier: 'new-user',
        origin: 'https://new.example',
        requestId: 'save-b',
        site: 'new.example',
        step: 'password'
      }
    })
    const rpc = vi.fn().mockResolvedValue(null)

    await dismissSensitivePrompt(staleOverlay, rpc, vi.fn())

    expect(rpc).not.toHaveBeenCalled()
    expect(getOverlayState().vaultSaveLogin?.requestId).toBe('save-b')
  })

  it('ignores a stale login-save cancellation after the same request changes phase', async () => {
    resetOverlayState()
    patchOverlayState({
      vaultSaveLogin: {
        identifier: '',
        origin: 'https://example.com',
        requestId: 'save-a',
        site: 'example.com',
        step: 'identifier'
      }
    })
    const staleOverlay = getOverlayState()

    patchOverlayState({
      vaultSaveLogin: {
        identifier: 'employee',
        origin: 'https://example.com',
        requestId: 'save-a',
        site: 'example.com',
        step: 'password'
      }
    })
    const rpc = vi.fn().mockResolvedValue(null)

    await dismissSensitivePrompt(staleOverlay, rpc, vi.fn())

    expect(rpc).not.toHaveBeenCalled()
    expect(getOverlayState().vaultSaveLogin).toMatchObject({
      identifier: 'employee',
      requestId: 'save-a',
      step: 'password'
    })
  })

  it('cancels the visible code prompt before a hidden login-save prompt', async () => {
    resetOverlayState()
    patchOverlayState({
      vaultCode: { hint: 'email', requestId: 'vault-code-1', site: 'example.com' },
      vaultSaveLogin: {
        identifier: 'employee',
        origin: 'https://example.com',
        requestId: 'vault-save-1',
        site: 'example.com',
        step: 'password'
      }
    })
    const rpc = vi.fn().mockResolvedValue(null)

    await dismissSensitivePrompt(getOverlayState(), rpc, vi.fn())

    expect(rpc).toHaveBeenCalledWith('vault.code.respond', { code: '', request_id: 'vault-code-1' })
    expect(getOverlayState().vaultCode).toBeNull()
    expect(getOverlayState().vaultSaveLogin?.requestId).toBe('vault-save-1')
  })

  it('clears a sudo overlay before a stale cancel RPC resolves', async () => {
    resetOverlayState()
    patchOverlayState({ sudo: { requestId: 'sudo-1' } })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys)

    expect(getOverlayState().sudo).toBeNull()
    expect(sys).toHaveBeenCalledWith('sudo cancelled')
    expect(rpc).toHaveBeenCalledWith('sudo.respond', { password: '', request_id: 'sudo-1' })
    await pending
  })

  it('clears a secret overlay before a stale cancel RPC resolves', async () => {
    resetOverlayState()
    patchOverlayState({ secret: { envVar: 'API_KEY', prompt: 'Enter API key', requestId: 'secret-1' } })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys)

    expect(getOverlayState().secret).toBeNull()
    expect(sys).toHaveBeenCalledWith('secret entry cancelled')
    expect(rpc).toHaveBeenCalledWith('secret.respond', { request_id: 'secret-1', value: '' })
    await pending
  })

  it('cancels a vault login prompt without sending credentials', async () => {
    resetOverlayState()
    patchOverlayState({
      vaultSaveLogin: {
        identifier: 'employee',
        origin: 'https://example.com',
        requestId: 'vault-save-1',
        site: 'example.com',
        step: 'password'
      }
    })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys)

    expect(getOverlayState().vaultSaveLogin).toBeNull()
    expect(sys).toHaveBeenCalledWith('login save cancelled')
    expect(rpc).toHaveBeenCalledWith('vault.save_login.respond', { login: '', request_id: 'vault-save-1' })
    await pending
  })

  it('cancels a one-time-code prompt without exposing the code', async () => {
    resetOverlayState()
    patchOverlayState({ vaultCode: { hint: 'email', requestId: 'vault-code-1', site: 'example.com' } })
    const rpc = vi.fn().mockResolvedValue(null)
    const sys = vi.fn()

    const pending = dismissSensitivePrompt(getOverlayState(), rpc, sys)

    expect(getOverlayState().vaultCode).toBeNull()
    expect(sys).toHaveBeenCalledWith('verification code entry cancelled')
    expect(rpc).toHaveBeenCalledWith('vault.code.respond', { code: '', request_id: 'vault-code-1' })
    await pending
  })
})
