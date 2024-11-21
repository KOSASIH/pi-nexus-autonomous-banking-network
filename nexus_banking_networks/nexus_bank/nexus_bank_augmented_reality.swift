import ARKit
import UIKit

class AugmentedRealityViewController: UIViewController, ARSCNViewDelegate {
  @IBOutlet var sceneView: ARSCNView!

  override func viewDidLoad() {
    super.viewDidLoad()
    sceneView.delegate = self
    sceneView.showsStatistics = true
  }

  func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
    // Add a 3D model to the scene
    let model = SCNScene(named: "model.scn")!
    node.addChildNode(model.rootNode)
  }
}
