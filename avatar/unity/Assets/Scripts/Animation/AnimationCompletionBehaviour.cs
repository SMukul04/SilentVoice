using UnityEngine;

namespace Avatar.Animation
{
    public class AnimationCompletionBehaviour : StateMachineBehaviour
    {
        private bool _completionSent = false;
        private AvatarAnimationPlayer _player;

        public override void OnStateEnter(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
        {
            _completionSent = false;
            
            if (_player == null)
            {
                _player = animator.GetComponent<AvatarAnimationPlayer>();
                if (_player == null)
                {
                    _player = animator.GetComponentInParent<AvatarAnimationPlayer>();
                }
            }
        }

        public override void OnStateUpdate(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
        {
            if (!_completionSent && stateInfo.normalizedTime >= 1.0f)
            {
                _completionSent = true;
                if (_player != null)
                {
                    _player.OnAnimationComplete();
                }
            }
        }
    }
}
